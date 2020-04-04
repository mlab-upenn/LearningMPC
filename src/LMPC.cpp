
#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Point.h>
#include <ackermann_msgs/AckermannDriveStamped.h>
#include <nav_msgs/OccupancyGrid.h>
#include <tf/transform_listener.h>

#include <math.h>
#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <random>
#include <LearningMPC/spline.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <nav_msgs/Odometry.h>

#include <LearningMPC/track.h>
#include <Eigen/Sparse>
#include "OsqpEigen/OsqpEigen.h"
#include <unsupported/Eigen/MatrixFunctions>

const int nx = 3;
const int nu = 2;
const double CAR_LENGTH = 0.35;

using namespace std;
using namespace Eigen;

struct Sample{
    Matrix<double,nx,1> x;
    Matrix<double,nu,1> u;
    double s;
    int time;
    int iter;
    int cost;
};
enum rviz_id{
    CENTERLINE,
    CENTERLINE_POINTS,
    CENTERLINE_SPLINE,
    PREDICTION,
    BORDERLINES,
    SAFE_SET,
    TERMINAL_CANDIDATE,
    DEBUG
};

class LMPC{
public:
    LMPC(ros::NodeHandle& nh);

private:
    ros::NodeHandle nh_;
    ros::Publisher track_viz_pub_;
    ros::Publisher trajectories_viz_pub_;
    ros::Publisher LMPC_viz_pub_;
    ros::Publisher drive_pub_;
    ros::Publisher debugger_pub_;

    ros::Subscriber odom_sub_;
    ros::Subscriber rrt_sub_;
    ros::Subscriber map_sub_;

    /*Paramaters*/
    string pose_topic;
    string drive_topic;
    string wp_file_name;
    double WAYPOINT_SPACE;
    double Ts;
    double ds;
    int speed_num;
    int steer_num;
    int N;
    int K_NEAR;
    double SPEED_MAX;
    double STEER_MAX;
    double ACCELERATION_MAX;
    double DECELERATION_MAX;
    double MAP_MARGIN;
    double SPEED_THRESHOLD;
    // MPC params
    double q_s;

    Track* track_;
    //odometry
    tf::Transform tf_;
    tf::Vector3 car_pos_;
    double yaw_;
    double s_prev_;
    double s_curr_;
    double speed_m_;

    //Sample Safe set
    vector<vector<Sample>> SS_;
    vector<Sample> curr_trajectory_;
    int iter_;
    int time_;
    Matrix<double,nx,1> terminal_state_pred_;

    // map info
    nav_msgs::OccupancyGrid map_;
    nav_msgs::OccupancyGrid map_updated_;

    VectorXd QPSolution_;
    bool first_run_;
    vector<geometry_msgs::Point> border_lines_;


    void getParameters(ros::NodeHandle& nh);
    void init_occupancy_grid();
    void init_SS_from_data(const string data_file);
    void visualize_centerline();
    int reset_QPSolution(int iter);
    void odom_callback(const nav_msgs::Odometry::ConstPtr &odom_msg);
    void add_point();
    void select_trajectory();
    void simulate_dynamics(Matrix<double,nx,1>& state, Matrix<double,nu,1>& input, double dt, Matrix<double,nx,1>& new_state);

    void solve_MPC(const Matrix<double,nx,1>& terminal_candidate);

    void get_linearized_dynamics(Matrix<double,nx,nx>& Ad, Matrix<double,nx, nu>& Bd, Matrix<double,nx,1>& hd,
            Matrix<double,nx,1>& x_op, Matrix<double,nu,1>& u_op);

    Vector3d global_to_track(double x, double y, double yaw, double s);
    Vector3d track_to_global(double e_y, double e_yaw, double s);

    void applyControl();
    void visualize_mpc_solution(const vector<Sample>& convex_safe_set, const Matrix<double,nx,1>& terminal_candidate);

    Matrix<double,nx,1> select_terminal_candidate();
    void select_convex_safe_set(vector<Sample>& convex_safe_set, int iter_start, int iter_end, double s);
    int find_nearest_point(vector<Sample>& trajectory, double s);
    void update_cost_to_go(vector<Sample>& trajectory);
    Matrix<double,nx,1> get_nonlinear_dynamics(Matrix<double,nx,1>& x, Matrix<double,nu,1>& u,  double t);
};

LMPC::LMPC(ros::NodeHandle &nh): nh_(nh){

    getParameters(nh_);
    init_occupancy_grid();
    track_ = new Track(wp_file_name, WAYPOINT_SPACE, map_, true);

    odom_sub_ = nh_.subscribe(pose_topic, 10, &LMPC::odom_callback, this);
    drive_pub_ = nh_.advertise<ackermann_msgs::AckermannDriveStamped>(drive_topic, 1);
   // rrt_sub_ = nh_.subscribe("path_found", 1, &LMPC::rrt_path_callback, this);
  //  map_sub_ = nh_.subscribe("map_updated", 1, &LMPC::map_callback, this);

    track_viz_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("track_centerline", 1);

    LMPC_viz_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("LMPC", 1);
    debugger_pub_ = nh_.advertise<visualization_msgs::Marker>("Debugger", 1);

    nav_msgs::Odometry odom_msg;
    boost::shared_ptr<nav_msgs::Odometry const> odom_ptr;
    odom_ptr = ros::topic::waitForMessage<nav_msgs::Odometry>("odom", ros::Duration(5));
    if (odom_ptr == nullptr){cout<< "fail to receive odom message!"<<endl;}
    else{
        odom_msg = *odom_ptr;
    }
    float x = odom_msg.pose.pose.position.x;
    float y = odom_msg.pose.pose.position.y;
    s_prev_ = track_->findTheta(x, y,0,true);
    car_pos_ = tf::Vector3(x, y, 0.0);
    yaw_ = tf::getYaw(odom_msg.pose.pose.orientation);

    iter_ = 2;

    init_SS_from_data("/home/yuwei/yuwei_ws/src/LearningMPC/initial_safe_set.csv");
    cout<<"SS size: "<<SS_.size()<<endl;
}

void LMPC::getParameters(ros::NodeHandle &nh) {
    nh.getParam("pose_topic", pose_topic);
    nh.getParam("drive_topic", drive_topic);
    nh.getParam("wp_file_name", wp_file_name);
    nh.getParam("N",N);
    nh.getParam("Ts",Ts);
    nh.getParam("K_NEAR", K_NEAR);
    nh.getParam("ACCELERATION_MAX", ACCELERATION_MAX);
    nh.getParam("DECELERATION_MAX", DECELERATION_MAX);
    nh.getParam("SPEED_MAX", SPEED_MAX);
    nh.getParam("STEER_MAX", STEER_MAX);
    nh.getParam("WAYPOINT_SPACE", WAYPOINT_SPACE);
//    nh.getParam("r_v",r_v);
//    nh.getParam("r_steer",r_steer);
    nh.getParam("q_s",q_s);
 //   R.diagonal() << r_v, r_steer;
    nh.getParam("MAP_MARGIN",MAP_MARGIN);
}

int compare_s(Sample& s1, Sample& s2){
    return (s1.s< s2.s);
}

void LMPC::init_occupancy_grid(){
    boost::shared_ptr<nav_msgs::OccupancyGrid const> map_ptr;
    map_ptr = ros::topic::waitForMessage<nav_msgs::OccupancyGrid>("map", ros::Duration(5.0));
    if (map_ptr == nullptr){ROS_INFO("No map received");}
    else{
        map_ = *map_ptr;
        map_updated_ = map_;
        ROS_INFO("Map received");
    }
    ROS_INFO("Initializing occupancy grid for map ...");
    occupancy_grid::inflate_map(map_, MAP_MARGIN);
}

void LMPC::init_SS_from_data(string data_file) {
    CSVReader reader(data_file);
    // Get the data from CSV File
    std::vector<std::vector<std::string>> dataList = reader.getData();
    SS_.clear();
    // Print the content of row by row on screen
    int time_prev=0;
    int it =0;
    vector<Sample> traj;
    for(std::vector<std::string> vec : dataList){
        Sample sample;
        sample.time = std::stof(vec.at(0));
        // check if it's a new lap
        if (sample.time - time_prev < 0) {
            it++;
            update_cost_to_go(traj);
            SS_.push_back(traj);
            traj.clear();
        }
        sample.x(0) = std::stof(vec.at(1));
        sample.x(1) = std::stof(vec.at(2));
        sample.x(2) = std::stof(vec.at(3));
        sample.u(0) = std::stof(vec.at(4));
        sample.u(1) = std::stof(vec.at(5));
        sample.s = std::stof(vec.at(6));
        sample.iter = it;
        traj.push_back(sample);
        time_prev = sample.time;
    }
    update_cost_to_go(traj);
    SS_.push_back(traj);
}

void LMPC::odom_callback(const nav_msgs::Odometry::ConstPtr &odom_msg){
    /******** process pose info *************/
    visualize_centerline();
    speed_m_ = odom_msg->twist.twist.linear.x;

    float x = odom_msg->pose.pose.position.x;
    float y = odom_msg->pose.pose.position.y;
    s_curr_ = track_->findTheta(x, y,0,true);
    car_pos_ = tf::Vector3(x, y, 0.0);
    yaw_ = tf::getYaw(odom_msg->pose.pose.orientation);

    if (first_run_){
        // initialize QPSolution_ from initial Sample Safe Set (using the 2nd iteration)
        reset_QPSolution(1);
        first_run_ = false;
    }

  /******** LMPC MAIN LOOP starts ********/

    /***check if it is new lap***/
    if (s_curr_ - s_prev_ < -track_->length/2){
        iter_++;
        update_cost_to_go(curr_trajectory_);
        sort(curr_trajectory_.begin(), curr_trajectory_.end(), compare_s);
        SS_.push_back(curr_trajectory_);
        curr_trajectory_.clear();
        reset_QPSolution(iter_-1);
        time_ = 0;
    }

    /*** select terminal state candidate and its convex safe set ***/
    Matrix<double,nx,1> terminal_candidate = select_terminal_candidate();
    /** solve MPC and record current state***/
    for (int i=0; i<5; i++){
        solve_MPC(terminal_candidate);
    }
    applyControl();
    add_point();
    /*** store info and advance to next time step***/
    terminal_state_pred_ = QPSolution_.segment<nx>(N*nx);
    s_prev_ = s_curr_;
    time_++;
}

void LMPC::visualize_centerline(){
    // plot waypoints
//    visualization_msgs::Marker dots;
//    dots.header.stamp = ros::Time::now();
//    dots.header.frame_id = "map";
//    dots.id = rviz_id::CENTERLINE;
//    dots.ns = "centerline";
//    dots.type = visualization_msgs::Marker::POINTS;
//    dots.scale.x = dots.scale.y = 0.08;
//    dots.scale.z = 0.04;
//    dots.action = visualization_msgs::Marker::ADD;
//    dots.pose.orientation.w = 1.0;
//    dots.color.r = 1.0;
//    dots.color.a = 1.0;
//    //dots.lifetime = ros::Duration();
//
//    vector<Point_ref> centerline_waypoints = track_->centerline_points;
//
//    for (int i=0; i<centerline_waypoints.size(); i++){
//        geometry_msgs::Point p;
//        p.x = centerline_waypoints.at(i).x;
//        p.y = centerline_waypoints.at(i).y;
//        dots.points.push_back(p);
//    }

    visualization_msgs::Marker spline_dots;
    spline_dots.header.stamp = ros::Time::now();
    spline_dots.header.frame_id = "map";
    spline_dots.id = rviz_id::CENTERLINE_SPLINE;
    spline_dots.ns = "centerline";
    spline_dots.type = visualization_msgs::Marker::LINE_STRIP;
    spline_dots.scale.x = spline_dots.scale.y = 0.02;
    spline_dots.scale.z = 0.02;
    spline_dots.action = visualization_msgs::Marker::ADD;
    spline_dots.pose.orientation.w = 1.0;
    spline_dots.color.b = 1.0;
    spline_dots.color.a = 1.0;
    // spline_dots.lifetime = ros::Duration();

    for (float t=0.0; t<track_->length; t+=0.05){
        geometry_msgs::Point p;
        p.x = track_->x_eval(t);
        p.y = track_->y_eval(t);
        spline_dots.points.push_back(p);
    }

    visualization_msgs::MarkerArray markers;
  //  markers.markers.push_back(dots);
    markers.markers.push_back(spline_dots);
    track_viz_pub_.publish(markers);
}

int LMPC::reset_QPSolution(int iter){
    QPSolution_ = VectorXd::Zero((N+1)*nx+ N*nu + nx*(N+1) + (2*K_NEAR+1));
    for (int i=0; i<N+1; i++){
        QPSolution_.segment<nx>(i*nx) = SS_[iter][i].x;
        if (i<N) QPSolution_.segment<nu>((N+1)*nx + i*nu) = SS_[iter][i].u;
    }
}

Matrix<double,nx,1> LMPC::select_terminal_candidate(){
    if (time_ == 0){
        return SS_.back()[N].x;
    }
    else{
        return terminal_state_pred_;
    }
}

void LMPC::add_point(){
    Sample point;
    point.x = Vector3d(car_pos_.x(), car_pos_.y(), yaw_);
    point.s = s_curr_;
    point.iter = iter_;
    point.time = time_;
    point.u = QPSolution_.segment<nu>((N+1)*nx);
    curr_trajectory_.push_back(point);
}

void LMPC::select_convex_safe_set(vector<Sample>& convex_safe_set, int iter_start, int iter_end, double s){
    for (int it = iter_start; it<= iter_end; it++){
        int nearest_ind = find_nearest_point(SS_[it], s);
        int start_ind, end_ind;
        bool overlap_with_finishing_line = false;
        int lap_cost = SS_[it][0].cost;

        if (K_NEAR%2 != 0 ) {
            start_ind = nearest_ind - (K_NEAR-1)/2;
            end_ind = nearest_ind + (K_NEAR-1)/2;
        }
        else{
            start_ind = nearest_ind - K_NEAR/2 + 1;
            end_ind = nearest_ind + K_NEAR/2;
        }

        vector<Sample> curr_set;
        if (end_ind > SS_[it].size()-1){ // front portion of set crossed finishing line
            for (int ind=start_ind; ind<SS_[it].size(); ind++){
                curr_set.push_back(SS_[it][ind]);
                // modify the cost-to-go for each point before finishing line
                // to incentivize the car to cross finishing line towards a new lap
                curr_set[curr_set.size()-1].cost += lap_cost;
            }
            for (int ind=0; ind<end_ind-SS_[it].size()+1; ind ++){
                curr_set.push_back(SS_[it][ind]);
            }
            if (curr_set.size()!=K_NEAR) throw;  // for debug
        }
        else if (start_ind < 0){  //  set crossed finishing line
            for (int ind=start_ind+SS_[it].size(); ind<SS_[it].size(); ind++){
                // modify the cost-to-go, same
                curr_set.push_back(SS_[it][ind]);
                curr_set[curr_set.size()-1].cost += lap_cost;
            }
            for (int ind=0; ind<end_ind+1; ind ++){
                curr_set.push_back(SS_[it][ind]);
            }
            if (curr_set.size()!=K_NEAR) throw;  // for debug
        }
        else {  // no overlapping with finishing line
            for (int ind=start_ind; ind<=end_ind; ind++){
                curr_set.push_back(SS_[it][ind]);
            }
        }
        convex_safe_set.insert(convex_safe_set.end(), curr_set.begin(), curr_set.end());
    }
}

int LMPC::find_nearest_point(vector<Sample>& trajectory, double s){
    // binary search to find closest point to a given s
    int low = 0; int high = trajectory.size()-1;
    while (low<=high){
        int mid = (low + high)/2;
        if (s == trajectory[mid].s) return mid;
        if (s < trajectory[mid].s) high = mid-1;
        else low = mid+1;
    }
    return abs(trajectory[low].s-s) < (abs(trajectory[high].s-s))? low : high;


}

void LMPC::update_cost_to_go(vector<Sample>& trajectory){
    trajectory[trajectory.size()-1].cost = 0;
    for (int i=trajectory.size()-2; i>=0; i--){
        trajectory[i].cost = trajectory[i+1].cost + 1;
    }
}

Vector3d LMPC::global_to_track(double x, double y, double yaw, double s){
    double x_proj = track_->x_eval(s);
    double y_proj = track_->y_eval(s);
    double e_y = sqrt((x-x_proj)*(x-x_proj) + (y-y_proj)*(y-y_proj));
    double dx_ds = track_->x_eval_d(s);
    double dy_ds = track_->y_eval_d(s);
    e_y = dx_ds*(y-y_proj) - dy_ds*(x-x_proj) >0 ? e_y : -e_y;
    double e_yaw = yaw - atan2(dy_ds, dx_ds);
    while(e_yaw > M_PI) e_yaw -= 2*M_PI;
    while(e_yaw < -M_PI) e_yaw += 2*M_PI;

    return Vector3d(e_y, e_yaw, s);
}

Vector3d LMPC::track_to_global(double e_y, double e_yaw, double s){
    double dx_ds = track_->x_eval_d(s);
    double dy_ds = track_->y_eval_d(s);
    Vector2d proj(track_->x_eval(s), track_->y_eval(s));
    Vector2d pos = proj + Vector2d(-dy_ds, dx_ds).normalized()*e_y;
    double yaw = e_yaw + atan2(dy_ds, dx_ds);
    return Vector3d(pos(0), pos(1), yaw);
}

void LMPC::get_linearized_dynamics(Matrix<double,nx,nx>& Ad, Matrix<double,nx, nu>& Bd, Matrix<double,nx,1>& hd,
        Matrix<double,nx,1>& x_op, Matrix<double,nu,1>& u_op){

    double yaw = x_op(2);
    double v = u_op(0);
    double steer = u_op(1);

    Vector3d dynamics, h;
    dynamics(0) = u_op(0)*cos(x_op(2));
    dynamics(1) = u_op(0)*sin(x_op(2));
    dynamics(2) = tan(u_op(1))*u_op(0)/CAR_LENGTH;


    Matrix<double,nx,nx> A, M12;
    Matrix<double,nx,nu> B;

    A <<   0.0, 0.0, -v*sin(yaw),
            0.0, 0.0,  v*cos(yaw),
            0.0, 0.0,      0.0;

    B <<   cos(yaw), 0.0,
            sin(yaw), 0.0,
            tan(steer)/CAR_LENGTH, v/(cos(steer)*cos(steer)*CAR_LENGTH);

    Matrix<double,nx+nx,nx+nx> aux, M;
    aux.setZero();
    aux.block<nx,nx>(0,0) << A;
    aux.block<nx,nx>(0, nx) << Matrix3d::Identity();
    M = (aux*Ts).exp();
    M12 = M.block<nx,nx>(0,nx);
    h = dynamics - (A*x_op + B*u_op);

    //Discretize with Euler approximation
    Ad = Matrix3d::Identity() + A*Ts;
    //Ad = (A*Ts).exp();
    Bd = Ts*B;
    hd = Ts*h;
    //Bd = M12*B;
    //hd = M12*h;
}

void wrap_angle(double& angle, const double angle_ref){
    while(angle - angle_ref > M_PI) {angle -= 2*M_PI;}
    while(angle - angle_ref < -M_PI) {angle += 2*M_PI;}
}

void LMPC::solve_MPC(const Matrix<double,nx,1>& terminal_candidate){
    vector<Sample> terminal_CSS;
    double s_t = track_->findTheta(terminal_candidate(0), terminal_candidate(1), 0, true);
    select_convex_safe_set(terminal_CSS, iter_-2, iter_-1, s_t);

    /** MPC variables: z = [x0, ..., xN, u0, ..., uN-1, s0, ..., sN, lambda0, ....., lambda(2*K_NEAR)]*
     *  constraints: dynamics, track bounds, input limits, acceleration limit, slack, lambdas, terminal state, sum of lambda's*/
    SparseMatrix<double> HessianMatrix((N+1)*nx+ N*nu + nx*(N+1) + (2*K_NEAR+1), (N+1)*nx+ N*nu + nx*(N+1)+ (2*K_NEAR+1));
    SparseMatrix<double> constraintMatrix((N+1)*nx+ 2*(N+1)*nx + N*nu + (N-1) + (N+1)*nx + (2*K_NEAR) + 4, (N+1)*nx+ N*nu + nx*(N+1)+ (2*K_NEAR+1));

    VectorXd gradient((N+1)*nx+ N*nu + (N+1)*nx + (2*K_NEAR+1));

    VectorXd lower((N+1)*nx+ 2*(N+1)*nx + N*nu + (N-1) + (N+1)*nx + (2*K_NEAR) + 4);
    VectorXd upper((N+1)*nx+ 2*(N+1)*nx + N*nu + (N-1) + (N+1)*nx + (2*K_NEAR) + 4);

    gradient.setZero();
    lower.setZero(); upper.setZero();

    Matrix<double,nx,1> x_k_ref;
    Matrix<double,nu,1> u_k_ref;
    Matrix<double,nx,nx> Ad;
    Matrix<double,nx,nu> Bd;
    Matrix<double,nx,1> x0, hd;
    border_lines_.clear();

    x0 <<car_pos_.x(), car_pos_.y(), yaw_;

    /** make sure there are no discontinuities in yaw**/
    // first check terminal safe_set
    for (int i=0; i<terminal_CSS.size(); i++){
        wrap_angle(terminal_CSS[i].x(2), x0(2));
    }
    // also check for previous QPSolution
    for (int i=0; i<N+1; i++){
        wrap_angle(QPSolution_(i*nx+2), x0(2));
    }

    for (int i=0; i<N+1; i++){        //0 to N

        x_k_ref = QPSolution_.segment<nx>(i*nx);
        u_k_ref = QPSolution_.segment<nu>((N+1)*nx + i*nu);
        double s_ref = track_->findTheta(x_k_ref(0), x_k_ref(1), 0, true);
        get_linearized_dynamics(Ad, Bd, hd, x_k_ref, u_k_ref);
        /* form Hessian entries*/
        // cost does not depend on x0, only 1 to N

        for (int row = 0; row < nx; row++) {
            HessianMatrix.insert((N+1)*nx+N*nu +i*nx+row, (N+1)*nx+N*nu +i*nx+row) = q_s;
        }

        /* form constraint matrix */
        if (i<N){
            // Ad
            for (int row=0; row<nx; row++){
                for(int col=0; col<nx; col++){
                    constraintMatrix.insert((i+1)*nx+row, i*nx+col) = Ad(row,col);
                }
            }
            // Bd
            for (int row=0; row<nx; row++){
                for(int col=0; col<nu; col++){
                    constraintMatrix.insert((i+1)*nx+row, (N+1)*nx+ i*nu+col) = Bd(row,col);
                }
            }
            lower.segment<nx>((i+1)*nx) = -hd;
            upper.segment<nx>((i+1)*nx) = -hd;
        }

        // -I for each x_k+1
        for (int row=0; row<nx; row++) {
            constraintMatrix.insert(i*nx+row, i*nx+row) = -1.0;
        }

        double dx_dtheta = track_->x_eval_d(s_ref);
        double dy_dtheta = track_->y_eval_d(s_ref);

        constraintMatrix.insert((N+1)*nx+ 2*i, i*nx) = -dy_dtheta;      // a*x
        constraintMatrix.insert((N+1)*nx+ 2*i, i*nx+1) = dx_dtheta;     // b*y
        constraintMatrix.insert((N+1)*nx+ 2*i, (N+1)*nx +N*nu +i) = 1.0;   // min(C1,C2) <= a*x + b*y + s_k <= inf

        constraintMatrix.insert((N+1)*nx+ 2*i+1, i*nx) = -dy_dtheta;      // a*x
        constraintMatrix.insert((N+1)*nx+ 2*i+1, i*nx+1) = dx_dtheta;     // b*y
        constraintMatrix.insert((N+1)*nx+ 2*i+1, (N+1)*nx +N*nu +i) = -1.0;   // -inf <= a*x + b*y - s_k <= max(C1,C2)

        //get upper line and lower line
        Vector2d left_tangent_p, right_tangent_p, center_p;
        Vector2d right_line_p1, right_line_p2, left_line_p1, left_line_p2;
        geometry_msgs::Point r_p1, r_p2, l_p1, l_p2;

        center_p << track_->x_eval(s_ref), track_->y_eval(s_ref);
        right_tangent_p = center_p + track_->getRightHalfWidth(s_ref) * Vector2d(dy_dtheta, -dx_dtheta).normalized();
        left_tangent_p  = center_p + track_->getLeftHalfWidth(s_ref) * Vector2d(-dy_dtheta, dx_dtheta).normalized();

        right_line_p1 = right_tangent_p + 0.15*Vector2d(dx_dtheta, dy_dtheta).normalized();
        right_line_p2 = right_tangent_p - 0.15*Vector2d(dx_dtheta, dy_dtheta).normalized();
        left_line_p1 = left_tangent_p + 0.15*Vector2d(dx_dtheta, dy_dtheta).normalized();
        left_line_p2 = left_tangent_p - 0.15*Vector2d(dx_dtheta, dy_dtheta).normalized();

        // For visualizing track boundaries
        r_p1.x = right_line_p1(0);  r_p1.y = right_line_p1(1);
        r_p2.x = right_line_p2(0);  r_p2.y = right_line_p2(1);
        l_p1.x = left_line_p1(0);   l_p1.y = left_line_p1(1);
        l_p2.x = left_line_p2(0);   l_p2.y = left_line_p2(1);
        border_lines_.push_back(r_p1);  border_lines_.push_back(r_p2);
        border_lines_.push_back(l_p1); border_lines_.push_back(l_p2);

        double C1 =  - dy_dtheta*right_tangent_p(0) + dx_dtheta*right_tangent_p(1);
        double C2 = - dy_dtheta*left_tangent_p(0) + dx_dtheta*left_tangent_p(1);

        lower((N+1)*nx+ 2*i) =  min(C1, C2);
        upper((N+1)*nx+ 2*i) = OsqpEigen::INFTY;

        lower((N+1)*nx+ 2*i+1) = -OsqpEigen::INFTY;
        upper((N+1)*nx+ 2*i+1) = max(C1, C2);

        // u_min < u < u_max
        if (i<N){
            for (int row=0; row<nu; row++){
                constraintMatrix.insert((N+1)*nx+ 2*(N+1)*nx +i*nu+row, (N+1)*nx+i*nu+row) = 1.0;
            }
            // input bounds: speed and steer
            lower.segment<nu>((N+1)*nx+ 2*(N+1)*nx +i*nu) <<  0.0, -STEER_MAX;
            upper.segment<nu>((N+1)*nx+ 2*(N+1)*nx +i*nu) <<SPEED_MAX, STEER_MAX;
        }

        if (i<N-1){
            //acceleration
            constraintMatrix.insert((N+1)*nx+ 2*(N+1)*nx + N*nu +i, (N+1)*nx+ i*nu) = -1;
            constraintMatrix.insert((N+1)*nx+ 2*(N+1)*nx + N*nu +i, (N+1)*nx+ (i+1)*nu) = 1;
            lower((N+1)*nx+(N+1)+(N+1)*nu+i) = -DECELERATION_MAX*Ts;
            upper((N+1)*nx+(N+1)+(N+1)*nu+i) = ACCELERATION_MAX*Ts;
        }

        // s_k >= 0
        for (int row=0; row<nx; row++){
            constraintMatrix.insert((N+1)*nx + 2*(N+1)*nx + N*nu + (N-1) + i*nx + row, (N+1)*nx+N*nu +i*nx+row) = 1.0;
            lower((N+1)*nx + 2*(N+1)*nx + N*nu  + (N-1) + i*nx + row) = 0;
            upper((N+1)*nx + 2*(N+1)*nx + N*nu  + (N-1) + i*nx + row) = OsqpEigen::INFTY;
        }
    }
    int numOfConstraintsSoFar = (N+1)*nx + 2*(N+1)*nx + N*nu + (N-1) + (N+1)*nx;

    // lamda's >= 0
    for (int i=0; i<2*K_NEAR; i++){
        constraintMatrix.insert(numOfConstraintsSoFar + i, (N+1)*nx+ N*nu + (N+1)*nx + i) = 1.0;
        lower(numOfConstraintsSoFar + i) = 0;
        upper(numOfConstraintsSoFar + i) = OsqpEigen::INFTY;
    }
    numOfConstraintsSoFar += 2*K_NEAR;

    // terminal state constraints: x_N+1 = linear_combination(lambda's)
    for (int i=0; i<2*K_NEAR; i++){
        for (int state_ind=0; state_ind<nx; state_ind++){
            constraintMatrix.insert(numOfConstraintsSoFar + state_ind, (N+1)*nx+ N*nu + (N+1)*nx + i) = terminal_CSS[i].x(state_ind);
        }
    }
    for (int state_ind=0; state_ind<nx; state_ind++){
        constraintMatrix.insert(numOfConstraintsSoFar + state_ind, N*nx + state_ind) = -1;
    }
    numOfConstraintsSoFar += nx;

    // sum of lamda's = 1;
    for (int i=0; i<2*K_NEAR; i++){
        constraintMatrix.insert(numOfConstraintsSoFar, (N+1)*nx+ N*nu + (N+1)*nx + i) = 1;
    }
    lower(numOfConstraintsSoFar) = 1.0;
    upper(numOfConstraintsSoFar) = 1.0;
    numOfConstraintsSoFar++;
    if (numOfConstraintsSoFar != (N+1)*nx+ 2*(N+1)*nx + N*nu + (N-1) + (N+1)*nx + (2*K_NEAR) +4) throw;  // for debug

    // gradient
    for (int i=0; i<2*K_NEAR; i++){
        gradient((N+1)*nx+ N*nu + (N+1)*nx + i) = terminal_CSS[i].cost;
    }

    //x0 constraint
    lower.head(nx) = -x0;
    upper.head(nx) = -x0;

    //v0 limit
    lower((N+1)*nx+ 2*(N+1)*nx) =  max(speed_m_-DECELERATION_MAX*Ts, 0.0);
    upper((N+1)*nx+ 2*(N+1)*nx) =  min(speed_m_+ACCELERATION_MAX*Ts, SPEED_MAX);

    SparseMatrix<double> H_t = HessianMatrix.transpose();
    SparseMatrix<double> sparse_I((N+1)*nx+ N*nu + nx*(N+1)+ (2*K_NEAR+1), (N+1)*nx+ N*nu + nx*(N+1)+ (2*K_NEAR+1));
    sparse_I.setIdentity();
    HessianMatrix = 0.5*(HessianMatrix + H_t) + 0.0000001*sparse_I;

    OsqpEigen::Solver solver;
    solver.settings()->setWarmStart(true);
    solver.data()->setNumberOfVariables((N+1)*nx+ N*nu + nx*(N+1)+ (2*K_NEAR+1));
    solver.data()->setNumberOfConstraints((N+1)*nx+ 2*(N+1)*nx + N*nu + (N-1) + (N+1)*nx + (2*K_NEAR) +4);

    if (!solver.data()->setHessianMatrix(HessianMatrix)) throw "fail set Hessian";
    if (!solver.data()->setGradient(gradient)){throw "fail to set gradient";}
    if (!solver.data()->setLinearConstraintsMatrix(constraintMatrix)) throw"fail to set constraint matrix";
    if (!solver.data()->setLowerBound(lower)){throw "fail to set lower bound";}
    if (!solver.data()->setUpperBound(upper)){throw "fail to set upper bound";}

    if(!solver.initSolver()){ cout<< "fail to initialize solver"<<endl;}

    if(!solver.solve()) {
        return;
    }
    QPSolution_ = solver.getSolution();
    cout<<"Solution: "<<endl;
    cout<<QPSolution_<<endl;
    solver.clearSolver();
    visualize_mpc_solution(terminal_CSS, terminal_candidate);

}

void LMPC::applyControl() {
    float speed = QPSolution_((N+1)*nx);
    float steer = QPSolution_((N+1)*nx+1);
    cout<<"steer_cmd: "<<steer<<endl;
    cout<<"speed_cmd: "<<speed<<endl;

    steer = min(steer, 0.41f);
    steer = max(steer, -0.41f);

    ackermann_msgs::AckermannDriveStamped ack_msg;
    ack_msg.drive.speed = speed;
    ack_msg.drive.steering_angle = steer;
    ack_msg.drive.steering_angle_velocity = 1.0;
    drive_pub_.publish(ack_msg);
}

void LMPC::visualize_mpc_solution(const vector<Sample>& convex_safe_set, const Matrix<double,nx,1>& terminal_candidate) {
    visualization_msgs::MarkerArray markers;

    visualization_msgs::Marker pred_dots;
    pred_dots.header.stamp = ros::Time::now();
    pred_dots.header.frame_id = "map";
    pred_dots.id = rviz_id::PREDICTION;
    pred_dots.ns = "predicted_positions";
    pred_dots.type = visualization_msgs::Marker::POINTS;
    pred_dots.scale.x = pred_dots.scale.y = pred_dots.scale.z = 0.05;
    pred_dots.action = visualization_msgs::Marker::ADD;
    pred_dots.pose.orientation.w = 1.0;
    pred_dots.color.g = 1.0;
    pred_dots.color.a = 1.0;
    for (int i=0; i<N+1; i++){
        geometry_msgs::Point p;
        p.x = QPSolution_(i*nx);
        p.y = QPSolution_(i*nx+1);
        pred_dots.points.push_back(p);
    }
    markers.markers.push_back(pred_dots);

    visualization_msgs::Marker borderlines;
    borderlines.header.stamp = ros::Time::now();
    borderlines.header.frame_id = "map";
    borderlines.id = rviz_id::BORDERLINES;
    borderlines.ns = "borderlines";
    borderlines.type = visualization_msgs::Marker::LINE_LIST;
    borderlines.scale.x = 0.03;
    borderlines.action = visualization_msgs::Marker::ADD;
    borderlines.pose.orientation.w = 1.0;
    borderlines.color.r = 1.0;
    borderlines.color.a = 1.0;

    borderlines.points = border_lines_;
    markers.markers.push_back(borderlines);

    visualization_msgs::Marker css_dots;
    css_dots.header.stamp = ros::Time::now();
    css_dots.header.frame_id = "map";
    css_dots.id = rviz_id::SAFE_SET;
    css_dots.ns = "safe_set";
    css_dots.type = visualization_msgs::Marker::POINTS;
    css_dots.scale.x = css_dots.scale.y = css_dots.scale.z = 0.04;
    css_dots.action = visualization_msgs::Marker::ADD;
    css_dots.pose.orientation.w = 1.0;
    css_dots.color.g = 1.0;
    css_dots.color.b = 1.0;
    css_dots.color.a = 1.0;
    VectorXd costs = VectorXd(convex_safe_set.size());
    for (int i=0; i<convex_safe_set.size(); i++){
        geometry_msgs::Point p;
        p.x = convex_safe_set[i].x(0);
        p.y = convex_safe_set[i].x(1);
        css_dots.points.push_back(p);
        costs(i) = convex_safe_set[i].cost;
    }
    cout<<"costs: "<<costs<< endl;
    markers.markers.push_back(css_dots);

    visualization_msgs::Marker terminal_dot;
    terminal_dot.header.stamp = ros::Time::now();
    terminal_dot.header.frame_id = "map";
    terminal_dot.id = rviz_id::TERMINAL_CANDIDATE;
    terminal_dot.ns = "terminal_candidate";
    terminal_dot.type = visualization_msgs::Marker::SPHERE;
    terminal_dot.scale.x = terminal_dot.scale.y = terminal_dot.scale.z = 0.1;
    terminal_dot.action = visualization_msgs::Marker::ADD;
    terminal_dot.pose.orientation.w = 1.0;
    terminal_dot.pose.position.x = terminal_candidate(0);
    terminal_dot.pose.position.y = terminal_candidate(1);
    terminal_dot.color.r = 0.5;
    terminal_dot.color.b = 0.8;
    terminal_dot.color.a = 1.0;
    markers.markers.push_back(terminal_dot);

    LMPC_viz_pub_.publish(markers);
}

int main(int argc, char **argv){
    ros::init(argc, argv, "LMPC");
    ros::NodeHandle nh;
    LMPC LMPC(nh);
    ros::Rate rate(20);
    while(ros::ok()){
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}