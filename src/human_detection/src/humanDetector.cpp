#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav2_msgs/action/navigate_to_pose.hpp>

#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <utility>
#include <queue>

using NavigateToPose = nav2_msgs::action::NavigateToPose;
using GoalHandleNavigateToPose = rclcpp_action::ClientGoalHandle<NavigateToPose>;

class HumanDetectionController : public rclcpp::Node
{
public:
  HumanDetectionController()
  : Node("human_detection_controller")
  {
    // QoS settings
    auto qos_map  = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
    auto qos_amcl = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().transient_local();
    auto qos_scan = rclcpp::SensorDataQoS();

    // Subscribers
    map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
      "/map", qos_map,
      std::bind(&HumanDetectionController::mapCallback, this, std::placeholders::_1));

    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan", qos_scan,
      std::bind(&HumanDetectionController::scanCallback, this, std::placeholders::_1));

    amcl_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/amcl_pose", qos_amcl,
      std::bind(&HumanDetectionController::amclPoseCallback, this, std::placeholders::_1));

    // Navigation action client
    nav_client_ = rclcpp_action::create_client<NavigateToPose>(this, "navigate_to_pose");

    // Timer for main control loop
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(500),
      std::bind(&HumanDetectionController::controlLoop, this));

    // State initialization
    state_ = State::WAITING_FOR_MAP;
    map_received_ = false;
    scan_count_ = 0;

    RCLCPP_INFO(this->get_logger(), "Human Detection Controller initialized");
    RCLCPP_INFO(this->get_logger(), "Waiting for map and initial pose...");
  }

private:
  enum class State {
    WAITING_FOR_MAP,
    FINDING_HUMANS_IN_MAP,
    NAVIGATING_TO_HUMAN,
    RECORDING_INITIAL_POSITIONS,
    MONITORING_FOR_CHANGES,
    INVESTIGATING_CHANGE,
    DONE
  };

  struct Point {
    double x, y;
    Point(double x_ = 0, double y_ = 0) : x(x_), y(y_) {}
    double distanceTo(const Point& other) const {
      double dx = x - other.x;
      double dy = y - other.y;
      return std::sqrt(dx * dx + dy * dy);
    }
  };

  // ==================== Callbacks ====================

  void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
  {
    if (!map_received_) {
      map_ = *msg;
      map_received_ = true;
      RCLCPP_INFO(this->get_logger(), "Map received: %dx%d, resolution: %.3f m/cell",
                  map_.info.width, map_.info.height, map_.info.resolution);
    }
  }

  void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
  {
    current_scan_ = *msg;
    scan_count_++;
  }

  void amclPoseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
  {
    current_pose_ = msg->pose.pose;
  }

  // ==================== Main Control Loop ====================

  void controlLoop()
  {
    switch (state_) {
      case State::WAITING_FOR_MAP:
        if (map_received_ && scan_count_ > 5) {
          RCLCPP_INFO(this->get_logger(), "Map and sensors ready. Analyzing map for humans...");
          state_ = State::FINDING_HUMANS_IN_MAP;
        }
        break;

      case State::FINDING_HUMANS_IN_MAP:
        findHumansInMap();
        break;

      case State::NAVIGATING_TO_HUMAN:
        // Waiting for navigation to complete
        break;

      case State::RECORDING_INITIAL_POSITIONS:
        recordInitialPosition();
        break;

      case State::MONITORING_FOR_CHANGES:
        monitorForChanges();
        break;

      case State::INVESTIGATING_CHANGE:
        // Waiting for investigation navigation
        break;

      case State::DONE:
        // Mission complete
        break;
    }
  }

  // ==================== Find Humans in Map ====================

  void findHumansInMap()
  {
    RCLCPP_INFO(this->get_logger(), "Scanning map for human obstacles...");
    
    // Find occupied cells that could be humans
    std::vector<Point> occupied_points;
    
    for (unsigned int y = 0; y < map_.info.height; ++y) {
      for (unsigned int x = 0; x < map_.info.width; ++x) {
        int index = y * map_.info.width + x;
        int8_t value = map_.data[index];
        
        // Occupied cells (value = 100)
        if (value > 50) {
          Point world_pt = gridToWorld(x, y);
          occupied_points.push_back(world_pt);
        }
      }
    }

    RCLCPP_INFO(this->get_logger(), "Found %zu occupied cells", occupied_points.size());

    // Cluster the occupied points to find human-sized obstacles
    human_locations_ = clusterOccupiedCells(occupied_points);

    if (human_locations_.empty()) {
      RCLCPP_WARN(this->get_logger(), "No human-like obstacles found in map!");
      state_ = State::DONE;
      return;
    }

    RCLCPP_INFO(this->get_logger(), "Found %zu potential human locations:", human_locations_.size());
    for (size_t i = 0; i < human_locations_.size(); ++i) {
      RCLCPP_INFO(this->get_logger(), "  Human %zu: (%.2f, %.2f)", 
                  i + 1, human_locations_[i].x, human_locations_[i].y);
    }

    // Start navigating to first human
    current_human_index_ = 0;
    navigateToHuman(current_human_index_);
  }

  std::vector<Point> clusterOccupiedCells(const std::vector<Point>& points)
  {
    if (points.empty()) return {};

    std::vector<Point> clusters;
    std::vector<bool> processed(points.size(), false);
    const double CLUSTER_RADIUS = 0.5; // meters - human-sized clusters

    for (size_t i = 0; i < points.size(); ++i) {
      if (processed[i]) continue;

      std::vector<Point> cluster_points;
      std::queue<size_t> to_process;
      to_process.push(i);
      processed[i] = true;

      while (!to_process.empty()) {
        size_t idx = to_process.front();
        to_process.pop();
        cluster_points.push_back(points[idx]);

        for (size_t j = 0; j < points.size(); ++j) {
          if (!processed[j] && points[idx].distanceTo(points[j]) < CLUSTER_RADIUS) {
            processed[j] = true;
            to_process.push(j);
          }
        }
      }

      // Only keep clusters of reasonable size (human-like)
      if (cluster_points.size() > 20 && cluster_points.size() < 1000) {
        Point centroid = calculateCentroid(cluster_points);
        clusters.push_back(centroid);
      }
    }

    return clusters;
  }

  Point calculateCentroid(const std::vector<Point>& points)
  {
    double sum_x = 0, sum_y = 0;
    for (const auto& pt : points) {
      sum_x += pt.x;
      sum_y += pt.y;
    }
    return Point(sum_x / points.size(), sum_y / points.size());
  }

  Point gridToWorld(unsigned int grid_x, unsigned int grid_y)
  {
    double world_x = map_.info.origin.position.x + (grid_x + 0.5) * map_.info.resolution;
    double world_y = map_.info.origin.position.y + (grid_y + 0.5) * map_.info.resolution;
    return Point(world_x, world_y);
  }

  // ==================== Navigation ====================

  void navigateToHuman(size_t human_index)
  {
    if (human_index >= human_locations_.size()) {
      RCLCPP_ERROR(this->get_logger(), "Invalid human index!");
      return;
    }

    Point target = human_locations_[human_index];
    
    // Navigate to a point near the human (not on top of them)
    Point robot_pos(current_pose_.position.x, current_pose_.position.y);
    double dx = target.x - robot_pos.x;
    double dy = target.y - robot_pos.y;
    double dist = std::sqrt(dx * dx + dy * dy);
    
    // Stop 1.5m away from human
    double approach_dist = std::max(0.0, dist - 1.5);
    Point nav_target(
      robot_pos.x + (dx / dist) * approach_dist,
      robot_pos.y + (dy / dist) * approach_dist
    );

    RCLCPP_INFO(this->get_logger(), "Navigating to human %zu at (%.2f, %.2f)...", 
                human_index + 1, nav_target.x, nav_target.y);

    if (!nav_client_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(this->get_logger(), "Navigation action server not available!");
      state_ = State::DONE;
      return;
    }

    auto goal_msg = NavigateToPose::Goal();
    goal_msg.pose.header.frame_id = "map";
    goal_msg.pose.header.stamp = this->now();
    goal_msg.pose.pose.position.x = nav_target.x;
    goal_msg.pose.pose.position.y = nav_target.y;
    goal_msg.pose.pose.orientation.w = 1.0;

    auto send_goal_options = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
    send_goal_options.result_callback = 
      std::bind(&HumanDetectionController::navigationResultCallback, this, std::placeholders::_1);

    nav_client_->async_send_goal(goal_msg, send_goal_options);
    state_ = State::NAVIGATING_TO_HUMAN;
  }

  void navigationResultCallback(const GoalHandleNavigateToPose::WrappedResult& result)
  {
    if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
      RCLCPP_INFO(this->get_logger(), "Navigation succeeded!");
      
      if (state_ == State::NAVIGATING_TO_HUMAN) {
        state_ = State::RECORDING_INITIAL_POSITIONS;
      } else if (state_ == State::INVESTIGATING_CHANGE) {
        recordNewPosition();
      }
    } else {
      RCLCPP_WARN(this->get_logger(), "Navigation failed!");
      state_ = State::DONE;
    }
  }

  // ==================== Recording Positions ====================

  void recordInitialPosition()
  {
    // Wait a moment for scan to stabilize
    rclcpp::sleep_for(std::chrono::seconds(1));
    
    // Use laser scan to refine human position
    Point refined_pos = refineHumanPositionFromScan();
    
    initial_human_positions_.push_back(refined_pos);
    
    RCLCPP_INFO(this->get_logger(), 
                "\n========================================\n"
                "HUMAN %zu INITIAL POSITION RECORDED:\n"
                "  x = %.2f m\n"
                "  y = %.2f m\n"
                "========================================",
                current_human_index_ + 1, refined_pos.x, refined_pos.y);

    // Move to next human
    current_human_index_++;
    if (current_human_index_ < human_locations_.size()) {
      navigateToHuman(current_human_index_);
    } else {
      RCLCPP_INFO(this->get_logger(), 
                  "\n========================================\n"
                  "ALL INITIAL POSITIONS RECORDED!\n"
                  "Now monitoring for changes...\n"
                  "========================================");
      state_ = State::MONITORING_FOR_CHANGES;
    }
  }

  Point refineHumanPositionFromScan()
  {
    // Find closest obstacle in laser scan
    double robot_x = current_pose_.position.x;
    double robot_y = current_pose_.position.y;
    double robot_yaw = getYawFromQuaternion(current_pose_.orientation);

    double min_dist = std::numeric_limits<double>::max();
    Point closest_point;

    for (size_t i = 0; i < current_scan_.ranges.size(); ++i) {
      float range = current_scan_.ranges[i];
      if (std::isnan(range) || std::isinf(range) || range < 0.1) continue;

      double angle = current_scan_.angle_min + i * current_scan_.angle_increment;
      double global_angle = robot_yaw + angle;
      
      double x = robot_x + range * std::cos(global_angle);
      double y = robot_y + range * std::sin(global_angle);
      
      if (range < min_dist) {
        min_dist = range;
        closest_point = Point(x, y);
      }
    }

    return closest_point;
  }

  // ==================== Monitoring ====================

  void monitorForChanges()
  {
    // Compare current laser scans with expected readings from map
    if (current_scan_.ranges.empty()) return;

    double robot_x = current_pose_.position.x;
    double robot_y = current_pose_.position.y;
    double robot_yaw = getYawFromQuaternion(current_pose_.orientation);

    int significant_changes = 0;
    std::vector<Point> change_locations;

    for (size_t i = 0; i < current_scan_.ranges.size(); i += 10) { // Sample every 10th beam
      float measured_range = current_scan_.ranges[i];
      if (std::isnan(measured_range) || std::isinf(measured_range)) continue;

      double angle = current_scan_.angle_min + i * current_scan_.angle_increment;
      double global_angle = robot_yaw + angle;

      // Get expected range from map
      double expected_range = raycastInMap(robot_x, robot_y, global_angle);
      
      // Check for significant discrepancy
      if (std::abs(measured_range - expected_range) > 0.3) { // 30cm threshold
        significant_changes++;
        
        double x = robot_x + measured_range * std::cos(global_angle);
        double y = robot_y + measured_range * std::sin(global_angle);
        change_locations.push_back(Point(x, y));
      }
    }

    if (significant_changes > 20) { // Significant environment change detected
      RCLCPP_WARN(this->get_logger(), 
                  "\n========================================\n"
                  "CHANGE DETECTED! Humans may have moved!\n"
                  "Significant discrepancies: %d\n"
                  "========================================",
                  significant_changes);
      
      auto clusters = clusterPoints(change_locations, 1.0);
      
      if (!clusters.empty()) {
        RCLCPP_INFO(this->get_logger(), "Investigating new positions...");
        new_human_positions_.clear();
        current_human_index_ = 0;
        investigation_targets_ = clusters;
        navigateToInvestigate(0);
      }
    }
  }

  double raycastInMap(double start_x, double start_y, double angle)
  {
    double max_range = 10.0;
    double step = map_.info.resolution;
    
    for (double r = 0; r < max_range; r += step) {
      double x = start_x + r * std::cos(angle);
      double y = start_y + r * std::sin(angle);
      
      int grid_x = (int)((x - map_.info.origin.position.x) / map_.info.resolution);
      int grid_y = (int)((y - map_.info.origin.position.y) / map_.info.resolution);
      
      if (grid_x < 0 || grid_x >= (int)map_.info.width || 
          grid_y < 0 || grid_y >= (int)map_.info.height) {
        return max_range;
      }
      
      int index = grid_y * map_.info.width + grid_x;
      if (map_.data[index] > 50) { // Occupied
        return r;
      }
    }
    
    return max_range;
  }

  std::vector<Point> clusterPoints(const std::vector<Point>& points, double radius)
  {
    if (points.empty()) return {};
    
    std::vector<Point> clusters;
    std::vector<bool> processed(points.size(), false);

    for (size_t i = 0; i < points.size(); ++i) {
      if (processed[i]) continue;

      double sum_x = points[i].x;
      double sum_y = points[i].y;
      int count = 1;
      processed[i] = true;

      for (size_t j = i + 1; j < points.size(); ++j) {
        if (processed[j]) continue;
        if (points[i].distanceTo(points[j]) < radius) {
          sum_x += points[j].x;
          sum_y += points[j].y;
          count++;
          processed[j] = true;
        }
      }

      clusters.push_back(Point(sum_x / count, sum_y / count));
    }

    return clusters;
  }

  void navigateToInvestigate(size_t target_index)
  {
    if (target_index >= investigation_targets_.size()) {
      state_ = State::DONE;
      return;
    }

    Point target = investigation_targets_[target_index];
    RCLCPP_INFO(this->get_logger(), "Investigating location (%.2f, %.2f)...", 
                target.x, target.y);

    auto goal_msg = NavigateToPose::Goal();
    goal_msg.pose.header.frame_id = "map";
    goal_msg.pose.header.stamp = this->now();
    goal_msg.pose.pose.position.x = target.x;
    goal_msg.pose.pose.position.y = target.y;
    goal_msg.pose.pose.orientation.w = 1.0;

    auto send_goal_options = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
    send_goal_options.result_callback = 
      std::bind(&HumanDetectionController::navigationResultCallback, this, std::placeholders::_1);

    nav_client_->async_send_goal(goal_msg, send_goal_options);
    state_ = State::INVESTIGATING_CHANGE;
  }

  void recordNewPosition()
  {
    rclcpp::sleep_for(std::chrono::seconds(1));
    Point new_pos = refineHumanPositionFromScan();
    new_human_positions_.push_back(new_pos);

    RCLCPP_INFO(this->get_logger(),
                "\n========================================\n"
                "HUMAN %zu NEW POSITION:\n"
                "  Initial: (%.2f, %.2f)\n"
                "  Current: (%.2f, %.2f)\n"
                "========================================",
                current_human_index_ + 1,
                initial_human_positions_[current_human_index_].x,
                initial_human_positions_[current_human_index_].y,
                new_pos.x, new_pos.y);

    current_human_index_++;
    if (current_human_index_ < investigation_targets_.size()) {
      navigateToInvestigate(current_human_index_);
    } else {
      RCLCPP_INFO(this->get_logger(), "\nInvestigation complete!");
      state_ = State::DONE;
    }
  }

  // ==================== Helpers ====================

  double getYawFromQuaternion(const geometry_msgs::msg::Quaternion& q)
  {
    double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    return std::atan2(siny_cosp, cosy_cosp);
  }

  // ==================== Members ====================

  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr amcl_pose_sub_;
  rclcpp_action::Client<NavigateToPose>::SharedPtr nav_client_;
  rclcpp::TimerBase::SharedPtr timer_;

  nav_msgs::msg::OccupancyGrid map_;
  sensor_msgs::msg::LaserScan current_scan_;
  geometry_msgs::msg::Pose current_pose_;

  State state_;
  bool map_received_;
  int scan_count_;
  
  std::vector<Point> human_locations_;
  std::vector<Point> initial_human_positions_;
  std::vector<Point> new_human_positions_;
  std::vector<Point> investigation_targets_;
  size_t current_human_index_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<HumanDetectionController>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}