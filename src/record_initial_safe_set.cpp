//
// Created by yuwei on 4/2/20.
//
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <ackermann_msgs/AckermannDriveStamped.h>
#include <LearningMPC/track.h>
#include <iostream>
#include <fstream>
#include <string>
#include <std_msgs/Float32.h>

using namespace std;

class Record_SS{
public:
    Record_SS(ros::NodeHandle& nh);

private:
    ros::NodeHandle nh_;
    ros::Subscriber cmd_sub_;
    ros::Subscriber odom_sub_;
    double s_prev_;
    int lap_;
    int time_;
    ofstream data_file_;
    double ros_time_prev_;
    nav_msgs::Odometry odom_;
    double acc_cmd_;


    Track* track_;
    void cmd_callback(const ackermann_msgs::AckermannDriveStampedConstPtr &cmd_msg);
    void odom_callback(const nav_msgs::Odometry::ConstPtr &odom_msg);
    void accel_cmd_callback(const std_msgs::Float32ConstPtr & accel_cmd);
};

Record_SS::Record_SS(ros::NodeHandle &nh) : nh_(nh) {
    cmd_sub_ = nh_.subscribe("nav", 1, &Record_SS::cmd_callback, this);
    odom_sub_ = nh_.subscribe("odom", 1, &Record_SS::odom_callback, this);

    string wp_file;
    double space;
    nh.getParam("waypoint_file", wp_file);
    nh.getParam("WAYPOINT_SPACE", space);

    boost::shared_ptr<nav_msgs::OccupancyGrid const> map_ptr;
    map_ptr = ros::topic::waitForMessage<nav_msgs::OccupancyGrid>("map", ros::Duration(5.0));
    if (map_ptr == nullptr){ROS_INFO("No map received");}
    else{
        ROS_INFO("Map received");
        nav_msgs::OccupancyGrid map = *map_ptr;
        track_ = new Track(wp_file, map, true);
    }

    nav_msgs::Odometry odom_msg;
    boost::shared_ptr<nav_msgs::Odometry const> odom_ptr;
    odom_ptr = ros::topic::waitForMessage<nav_msgs::Odometry>("odom", ros::Duration(5));
    if (odom_ptr == nullptr){cout<< "fail to receive odom message!"<<endl;}
    else{
        odom_msg = *odom_ptr;
    }
    float x = odom_msg.pose.pose.position.x;
    float y = odom_msg.pose.pose.position.y;
    s_prev_ = track_->findTheta(x,y,0,true);

    ros_time_prev_ = ros::Time::now().toSec();
    time_=0; lap_=0;

    data_file_.open( "initial_safe_set.csv");
}

void Record_SS::odom_callback(const nav_msgs::Odometry::ConstPtr &odom_msg){
    odom_ = *odom_msg;
}

void Record_SS::accel_cmd_callback(const std_msgs::Float32ConstPtr & accel_cmd){
    acc_cmd_ = accel_cmd->data;
}

void Record_SS::cmd_callback(const ackermann_msgs::AckermannDriveStampedConstPtr &cmd_msg) {

    if (cmd_msg != nullptr){
        cout<<"recording: "<<time_<<endl;
        double speed_cmd = cmd_msg->drive.speed;
        double steer_cmd = cmd_msg->drive.steering_angle;

        double yaw = tf::getYaw(odom_.pose.pose.orientation);
        float x = odom_.pose.pose.position.x;
        float y = odom_.pose.pose.position.y;
        double s_curr = track_->findTheta(x,y,0,true);
        double vel = odom_.twist.twist.linear.x;

        // check if is a new lap;
        if (s_curr - s_prev_ < -track_->length/2){
//            cout<<"s_curr: "<<s_curr<<endl;
//            cout<<"s_prev: "<<s_prev_<<endl;
            time_ = 0;
            lap_++;
            if (lap_>1){ //initial two laps completed
                data_file_.close();
                delete(track_);
                ros::shutdown();
            }
        }
        data_file_ << time_ <<","<<x<< ","<<y<<","<<yaw<<","<<vel<<","<<acc_cmd_<<","<<steer_cmd<<","<<s_curr<<endl;
        time_++;

        s_prev_ = s_curr;
        double ros_time_curr = ros::Time::now().toSec();
        cout << "dt: "<< ros_time_curr - ros_time_prev_ <<endl;
        ros_time_prev_ = ros_time_curr;
    }
}

int main(int argc, char **argv){
    ros::init(argc, argv, "record_init_ss");
    ros::NodeHandle nh;
    Record_SS rec_ss(nh);
    ros::spin();
    return 0;
}