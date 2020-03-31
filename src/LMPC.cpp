
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
#include <LearningMPC/occupancy_grid.h>
#include <unsupported/Eigen/MatrixFunctions>

const int nx = 3;
const int nu = 2;
const double CAR_LENGTH = 0.35;

using namespace std;
using namespace Eigen;

struct Sample{
    Matrix<double,nx,1> x;
    Matrix<double,nu,1> u;
    double time;
    int iter;
    int cost;
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

    Track track_;
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

    void getParameters(ros::NodeHandle& nh);
    void init_occupancy_grid();
    int reset_QPSolution(int iter);
    void odom_callback(const nav_msgs::Odometry::ConstPtr &odom_msg);
    void add_point();
    void select_trajectory();
    void simulate_dynamics(Matrix<double,nx,1>& state, Matrix<double,nu,1>& input, double dt, Matrix<double,nx,1>& new_state);

    void solve_MPC(Matrix<double,nx,1>& terminal_candidate);

    void get_linearized_dynamics(Matrix<double,nx,nx>& Ad, Matrix<double,nx, nu>& Bd, Matrix<double,nx,1> hd,
                                 Matrix<double,nx,1>& x_op, Matrix<double,nu,1>& u_op, double t);

    Vector3d global_to_track(double x, double y, double yaw, double s);
    Vector3d track_to_global(double e_y, double e_yaw, double s);

    void applyControl();

    Matrix<double,nx,1> select_terminal_candidate();
    void select_convex_safe_set(vector<int>& convex_safe_set_costs, vector<Matrix<double,nx,1>>& convex_safe_set, int iter_start, int iter_end, double s);
   // void select_evolved_convex_safe_set(vector<int>& evolved_convex_safe_set, vector<Matrix<double,nx,1>>& nearest_points, vector<Sample>& trajectory, double current_s);
    int find_nearest_point(vector<Sample>& trajectory, double s);
    void update_cost_to_go();
    Matrix<double,nx,1> get_nonlinear_dynamics(Matrix<double,nx,1>& x, Matrix<double,nu,1>& u,  double t);
};

int compare_s(Sample& s1, Sample& s2){
    return (s1.x(nx-1)< s2.x(nx-1));
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

void LMPC::odom_callback(const nav_msgs::Odometry::ConstPtr &odom_msg){
    /******** process pose info *************/
    speed_m_ = odom_msg->twist.twist.linear.x;

    tf::Quaternion q_tf;
    tf::quaternionMsgToTF(odom_msg->pose.pose.orientation, q_tf);
    tf::Matrix3x3 rot(q_tf);
    double roll, pitch;
    rot.getRPY(roll, pitch, yaw_);
    car_pos_ = tf::Vector3(odom_msg->pose.pose.position.x, odom_msg->pose.pose.position.y, 0.0);
    tf_.setOrigin(car_pos_);
    tf_.setRotation(q_tf);

    double s_curr_ = track_.findTheta(car_pos_.x(), car_pos_.y(), 0, true);
    if (first_run_){
        s_prev_ = s_curr_;
        // initialize QPSolution_ from initial Sample Safe Set (using the 2nd iteration)
        reset_QPSolution(1);
        first_run_ = false;
    }

  /******** LMPC MAIN LOOP starts ********/

    /***check if it is new lap***/
    if (s_curr_ - s_prev_ < -track_.length/2){
        iter_++;
        update_cost_to_go();
        sort(curr_trajectory_.begin(), curr_trajectory_.end(), compare_s);
        SS_.push_back(curr_trajectory_);
        curr_trajectory_.clear();
        reset_QPSolution(iter_-1);
        time_ = 0;
    }

    /*** select terminal state candidate and its convex safe set ***/
    Matrix<double,nx,1> terminal_candidate = select_terminal_candidate();
    /** solve MPC and record current state***/
    solve_MPC(terminal_candidate);
    applyControl();
    add_point();
    /*** store info and advance to next time step***/
    terminal_state_pred_ = QPSolution_.segment<nx>(N*nx);
    s_prev_ = s_curr_;
    time_++;
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
    point.x = global_to_track(car_pos_.x(), car_pos_.y(), yaw_, s_curr_);
    point.iter = iter_;
    point.time = time_;
    point.u = QPSolution_.segment<nu>((N+1)*nx);
    curr_trajectory_.push_back(point);
}

void LMPC::select_convex_safe_set(vector<int>& convex_safe_set_costs, vector<Matrix<double,nx,1>>& convex_safe_set, int iter_start, int iter_end, double s){
    for (int it = iter_start; it<= iter_end; it++){
        int nearest_ind = find_nearest_point(SS_[it], s);
        if (K_NEAR%2 != 0 ) {
            int start_ind = nearest_ind - (K_NEAR-1)/2;
            int end_ind = nearest_ind + (K_NEAR-1)/2;
        }
        else{
            int start_ind = nearest_ind - K_NEAR/2 + 1;
            int end_ind = nearest_ind + K_NEAR/2;
        }
        vector<Matrix<double,nx,1>> curr_set;

    }
}

int LMPC::find_nearest_point(vector<Sample>& trajectory, double s){
    // binary search to find closest point to a given s
    int low = 0; int high = trajectory.size()-1;
    while (low<=high){
        int mid = (low + high)/2;
        if (s == trajectory[mid].x(nx-1)) return mid;
        if (s < trajectory[mid].x(nx-1)) high = mid-1;
        else low = mid+1;
    }
    return abs(trajectory[low].x(nx-1)-s) < (abs(trajectory[high].x(nx-1)-s))? low : high;

}

void LMPC::update_cost_to_go(){
    curr_trajectory_[curr_trajectory_.size()-1].cost = 0;
    for (int i=curr_trajectory_.size()-2; i>=0; i--){
        curr_trajectory_[i].cost = curr_trajectory_[i+1].cost + 1;
    }

}

Vector3d LMPC::global_to_track(double x, double y, double yaw, double s){
    double x_proj = track_.x_eval(s);
    double y_proj = track_.y_eval(s);
    double e_y = sqrt((x-x_proj)*(x-x_proj) + (y-y_proj)*(y-y_proj));
    double dx_ds = track_.x_eval_d(s);
    double dy_ds = track_.y_eval_d(s);
    e_y = dx_ds*(y-y_proj) - dy_ds*(x-x_proj) >0 ? e_y : -e_y;
    double e_yaw = yaw - atan2(dy_ds, dx_ds);
    while(e_yaw > M_PI) e_yaw -= 2*M_PI;
    while(e_yaw < -M_PI) e_yaw += 2*M_PI;

    return Vector3d(e_y, e_yaw, s);
}

Vector3d LMPC::track_to_global(double e_y, double e_yaw, double s){
    double dx_ds = track_.x_eval_d(s);
    double dy_ds = track_.y_eval_d(s);
    Vector2d proj(track_.x_eval(s), track_.y_eval(s));
    Vector2d pos = proj + Vector2d(-dy_ds, dx_ds).normalized()*e_y;
    double yaw = e_yaw + atan2(dy_ds, dx_ds);
    return Vector3d(pos(0), pos(1), yaw);
}

Matrix<double,nx,1> LMPC::get_nonlinear_dynamics(Matrix<double,nx,1>& x, Matrix<double,nu,1>& u, double t){
    /* states: e_y, e_yaw, s
     * inputs: K, v
     */
    Matrix<double, nx,1> xdot;
    double dx_ds = track_.x_eval_d(t);
    double dy_ds = track_.y_eval_d(t);
    double d2x_ds2 = track_.x_eval_dd(t);
    double d2y_ds2 = track_.y_eval_dd(t);

    double numer = dx_ds*d2y_ds2 - dy_ds*d2x_ds2;
    double denom = pow(dx_ds,2) + pow(dy_ds, 2);

    double dphi_ds = numer/denom;

    //cout << "dphi_ds : "<< dphi_ds<<endl;

    double rho_s = track_.getCenterlineRadius(t);
    double Ks = track_.getCenterlineCurvature(t);

    xdot(0) = (rho_s - x(0))*tan(x(1))/rho_s;
    xdot(1) = (rho_s - x(0))*u(0)/(rho_s*cos(x(1))) - dphi_ds;
    xdot(2) = u(1)*cos(x(1))/(1-Ks*x(0));

    while(xdot(1) > M_PI)  xdot(1) -= 2*M_PI;
    while(xdot(1) < -M_PI) xdot(1) += 2*M_PI;

    cout<< "xdot: "<<endl;
    cout<<xdot<<endl;

    return xdot;
}

void LMPC::get_linearized_dynamics(Matrix<double,nx,nx>& Ad, Matrix<double,nx, nu>& Bd, Matrix<double,nx,1> hd,
                                    Matrix<double,nx,1>& x_op, Matrix<double,nu,1>& u_op, double t){

    Matrix<double,nx,nx> A, M12;
    Matrix<double,nx,nu> B;
    Matrix<double,nx,1> h;
    double rho_s = track_.getCenterlineRadius(t);
    double Ks = track_.getCenterlineCurvature(t);

    A <<   0.0,                 (rho_s-x_op(0))/rho_s,      0.0,
            -u_op(0)/rho_s,                        0.0,      0.0,
            - Ks*u_op(1)/max(0.01,x_op(2)),                          0.0,      -(1-x_op(0)*Ks)*u_op(1)/max(0.01,x_op(2)*x_op(2));

    B <<    0.0, 0.0,
            (rho_s-x_op(0))/rho_s, 0.0,
            0.0, (1-x_op(0)*Ks)/max(0.01,x_op(2));

    //Discretize with Euler approximation


    h = get_nonlinear_dynamics(x_op, u_op, t) - (A*x_op + B*u_op);

    Ad = (A*ds).exp();
    if(A.determinant()>0.01) {
        Bd = A.inverse()*(Ad - Matrix3d::Identity())*B;
        hd = A.inverse()*(Ad - Matrix3d::Identity())*h;
    }
    else {
        Matrix<double,nx+nx,nx+nx> aux, M;
        aux.setZero();
        aux.block<nx,nx>(0,0) << A;
        aux.block<nx,nx>(0, nx) << Matrix3d::Identity();
        M = (aux*ds).exp();
        M12 = M.block<nx,nx>(0,nx);
        Bd = M12 * B;
        hd = M12 * h;
    }
    cout<<"Ad: "<<endl;
    cout<<Ad<<endl;
    cout<<"Bd: "<<endl;
    cout<<Bd<<endl;
//   cout<<"hd: "<<endl;
    //cout<<hd<<endl;
}

void LMPC::solve_MPC(Matrix<double,nx,1>& terminal_candidate){
    vector<int> terminal_CSS_costs;
    vector<Matrix<double,nx,1>> terminal_CSS;
    select_convex_safe_set(terminal_CSS_costs, terminal_CSS, iter_-2, iter_-1, terminal_candidate(nx-1));

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

    x0 << global_to_track(car_pos_.x(), car_pos_.y(), yaw_, s_curr_);

    for (int i=0; i<N+1; i++){        //0 to N
        double Ks = track_.getCenterlineCurvature(s_curr_);

        x_k_ref = QPSolution_.segment<nx>(i*nx);
        u_k_ref = QPSolution_.segment<nu>((N+1)*nx + i*nu);
        double s_ref = x_k_ref(nx-1);
        get_linearized_dynamics(Ad, Bd, hd, x_k_ref, u_k_ref, s_curr_);
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
        /* track boundary constraints */
        for (int row=0; row<nx; row++) {
            //  -inf <= x_k - s_k <= x_max
            constraintMatrix.insert((N+1)*nx + i*nx+row, i*nx+row) = 1.0;
            constraintMatrix.insert((N+1)*nx + i*nx+row, (N+1)*nx+ N*nu + i*nx + row) = -1.0;
        }
        upper.segment<nx>((N+1)*nx + i*nx) << 0.4, M_PI/4.0, OsqpEigen::INFTY; //track_.getLeftHalfWidth(theta), M_PI/4.0;
        lower.segment<nx>((N+1)*nx + i*nx) << -OsqpEigen::INFTY, -OsqpEigen::INFTY, -OsqpEigen::INFTY;

        for (int row=0; row<nx; row++) {
            //  x_min <= x_k + s_k <= inf
            constraintMatrix.insert((N+1)*nx + (N+1)*nx +i*nx+row, i*nx+row) = 1.0;
            constraintMatrix.insert((N+1)*nx + (N+1)*nx +i*nx+row, (N+1)*nx+ N*nu + i*nx + row) = 1.0;
        }
        upper.segment<nx>((N+1)*nx + (N+1)*nx + i*nx) << OsqpEigen::INFTY, OsqpEigen::INFTY, OsqpEigen::INFTY; //track_.getLeftHalfWidth(theta), M_PI/4.0;
        lower.segment<nx>((N+1)*nx + (N+1)*nx + i*nx) << -0.4, -M_PI/4.0, 0.0; //-track_.getRightHalfWidth(theta), -M_PI/4.0;

        // u_min < u < u_max
        if (i<N){
            for (int row=0; row<nu; row++){
                constraintMatrix.insert((N+1)*nx+ 2*(N+1)*nx +i*nu+row, (N+1)*nx+i*nu+row) = 1.0;
            }
            // input bounds: speed and steer
            lower.segment<nu>((N+1)*nx+ 2*(N+1)*nx +i*nu) << -tan(STEER_MAX)/CAR_LENGTH, 0.0;
            upper.segment<nu>((N+1)*nx+ 2*(N+1)*nx +i*nu) << tan(STEER_MAX)/CAR_LENGTH, SPEED_MAX;
        }

        if (i<N-1){
            //acceleration
            constraintMatrix.insert((N+1)*nx+ 2*(N+1)*nx + N*nu +i, (N+1)*nx+ i*nu +1) = -1;
            constraintMatrix.insert((N+1)*nx+ 2*(N+1)*nx + N*nu +i, (N+1)*nx+ (i+1)*nu +1) = 1;
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
            constraintMatrix.insert(numOfConstraintsSoFar + state_ind, (N+1)*nx+ N*nu + (N+1)*nx + i) = terminal_CSS[i](state_ind);
        }
    }
    for (int state_ind=0; state_ind<nx; state_ind++){
        constraintMatrix.insert(numOfConstraintsSoFar + state_ind, N*nx + state_ind) = -1;
    }
    numOfConstraintsSoFar += nx;

    // sum of lamda's = 1;
    for (int i=0; i<2*K_NEAR; i++){
        constraintMatrix.insert(numOfConstraintsSoFar +1, (N+1)*nx+ N*nu + (N+1)*nx + i) = 1;
    }
    lower(numOfConstraintsSoFar + 1) = 1.0;
    upper(numOfConstraintsSoFar + 1) = 1.0;
    numOfConstraintsSoFar++;

    // gradient
    for (int i=0; i<2*K_NEAR; i++){
        gradient((N+1)*nx+ N*nu + (N+1)*nx + i) = terminal_CSS_costs[i];
    }

    //x0 constraint
    lower.head(nx) = -x0;
    upper.head(nx) = -x0;

    //v0 limit
    lower((N+1)*nx+ 2*(N+1)*nx + 1) =  max(speed_m_-DECELERATION_MAX*Ts, 0.0);
    upper((N+1)*nx+ 2*(N+1)*nx + 1) =  min(speed_m_+ACCELERATION_MAX*Ts, SPEED_MAX);

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
}

void LMPC::applyControl() {
    double u0 = QPSolution_((N+1)*nx);
    double u1 = QPSolution_((N+1)*nx+1);
    float steer = atan(u0*CAR_LENGTH);
    float speed = u1;
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