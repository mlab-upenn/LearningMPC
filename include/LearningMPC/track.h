//
// Created by yuwei on 11/20/19.
//

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

// standard
#include <math.h>
#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <algorithm>
#include <random>
#include <LearningMPC/spline.h>
#include <LearningMPC/CSVReader.h>
#include <Eigen/Dense>
#include <nav_msgs/OccupancyGrid.h>
#include <LearningMPC/occupancy_grid.h>

const int SEARCH_RANGE = 10;
const double HALF_WIDTH_MAX = 0.8;
using namespace std;

typedef struct Point_ref{
    double x;
    double y;
    double theta;  // theta represents progress along centerline_points
    double left_half_width;
    double right_half_width;
}Point_ref;

class Track{
public:
    vector<Point_ref> centerline_points;
    tk::spline X_spline;
    tk::spline Y_spline;
    double length;
    double space;
    nav_msgs::OccupancyGrid map;
    vector<double> width_info;

    Track(string file_name, nav_msgs::OccupancyGrid& map, bool sparse=false) : space(0.05), map(map){
        space = 0.05;
        centerline_points.clear();
        vector<geometry_msgs::Point> waypoints;
        CSVReader reader(file_name);
        // Get the data from CSV File
        std::vector<std::vector<std::string> > dataList = reader.getData();
        // Print the content of row by row on screen
        for(std::vector<std::string> vec : dataList){
            geometry_msgs::Point wp;
            wp.x = std::stof(vec.at(0));
            wp.y = std::stof(vec.at(1));
            waypoints.push_back(wp);
        }
        /*** process raw waypoints data, extract equally spaced points ***/
        int curr = 0;
        int next =1;
        Point_ref p_start;
        p_start.x = waypoints.at(0).x; p_start.y = waypoints.at(0).y; p_start.theta = 0.0;
        centerline_points.push_back(p_start);
        float theta = 0.0;

        while(next < waypoints.size()){
            float dist = sqrt(pow(waypoints.at(next).x-waypoints.at(curr).x, 2)
                    +pow(waypoints.at(next).y-waypoints.at(curr).y, 2));
            float dist_to_start = sqrt(pow(waypoints.at(next).x-waypoints.at(0).x, 2)
                                       +pow(waypoints.at(next).y-waypoints.at(0).y, 2));
            if (dist>space){
                theta += dist;
                Point_ref p;
                p.x = waypoints.at(next).x; p.y = waypoints.at(next).y; p.theta = theta;
                p.left_half_width = p.right_half_width = HALF_WIDTH_MAX;
                centerline_points.push_back(p);
                curr = next;
            }
            next++;
            // terminate when finished a lap
            if (next > waypoints.size()/2 && dist_to_start<space){
                break;
            }
        }
        double last_space = sqrt(pow(centerline_points.back().x-waypoints.at(0).x, 2)
                                 +pow(centerline_points.back().y-waypoints.at(0).y, 2));
        length = theta + last_space;
        Point_ref p_last, p_second_last;
        p_last.x = waypoints.at(0).x; p_last.y = waypoints.at(0).y; p_last.theta = length;

        p_second_last.x = 0.5*(centerline_points.back().x + p_last.x);
        p_second_last.y = 0.5*(centerline_points.back().y + p_last.y);
        p_second_last.theta = length - 0.5*last_space;

        if(last_space > space) {
            centerline_points.push_back(p_second_last);
        }
        centerline_points.push_back(p_last);   //close the loop

        vector<double> X;
        vector<double> Y;
        vector<double> thetas;
        for (int i=0; i<centerline_points.size(); i++){
            X.push_back(centerline_points.at(i).x);
            Y.push_back(centerline_points.at(i).y);
            thetas.push_back(centerline_points.at(i).theta);
        }

        X_spline.set_points(thetas, X);
        Y_spline.set_points(thetas, Y);

        /** if sparse, reinitialize centerline points such that they are densely and equally spaced **/
        if (sparse) {
            double s=0;
            centerline_points.clear();
            while (s<length){
                Point_ref p;
                p.x = X_spline(s);
                p.y = Y_spline(s);
                p.theta = s;
                s += space;
                centerline_points.push_back(p);
            }
        }
        initialize_width();
    }

    void initialize_width(){
        using namespace Eigen;
        int t=0;
        Vector2d p_right, p_left;
        for (int i=0; i<centerline_points.size(); i++){
            double dx_dtheta = x_eval_d(centerline_points[i].theta);
            double dy_dtheta = y_eval_d(centerline_points[i].theta);
            Vector2d p(centerline_points[i].x, centerline_points[i].y);

            //search right until hit right track boundary
            while(true){
                double x = (p + t*map.info.resolution*Vector2d(dy_dtheta, -dx_dtheta).normalized())(0);
                double y = (p + t*map.info.resolution*Vector2d(dy_dtheta, -dx_dtheta).normalized())(1);
                if(occupancy_grid::is_xy_occupied(map, x, y)){
                    p_right(0) = x;
                    p_right(1) = y;
                    break;
                }
                t++;
            }
            t=0;
            //search left until hit right track boundary
            while(true){
                double x = (p + t*map.info.resolution*Vector2d(-dy_dtheta, dx_dtheta).normalized())(0);
                double y = (p + t*map.info.resolution*Vector2d(-dy_dtheta, dx_dtheta).normalized())(1);
                if(occupancy_grid::is_xy_occupied(map, x, y)){
                    p_left(0) = x;
                    p_left(1) = y;
                    break;
                }
                t++;
            }
            centerline_points[i].left_half_width = (p_left-p).norm();
            centerline_points[i].right_half_width = (p_right-p).norm();
        }
    }

    double findTheta(double x, double y, double theta_guess, bool global_search= false){
        /* return: projected theta along centerline_points, theta is between [0, length]
         * */
          //  wrapTheta(theta_guess);
//            int start, end;
//            if(global_search){
//                start = 0;
//                end = centerline_points.size();
//            }
//            else{
//                int mid = int(floor(theta_guess/space));
//                start = mid - SEARCH_RANGE;
//                end = mid + SEARCH_RANGE;
//            }
//            int min_ind, second_min_ind;
//            double min_dist2 = 10000000.0;
//            for (int i=start; i<end; i++){
//                if (i>centerline_points.size()-1){ i=0;}
//                if (i<0){ i=centerline_points.size()-1;}
//                double dist2 = pow(x-centerline_points.at(i).x, 2) + pow(y-centerline_points.at(i).y, 2);
//                if( dist2 < min_dist2){
//                    min_dist2 = dist2;
//                    min_ind = i;
//                }
//            }
//            if (sqrt(min_dist2)>HALF_WIDTH_MAX && !global_search){
//                return findTheta(x, y, theta_guess, true);
//            }
//            Eigen::Vector2d p, p0, p1;
//            int min_ind_prev = min_ind-1;
//            int min_ind_next = min_ind+1;
//            if (min_ind_next>centerline_points.size()-1){ min_ind_next -= centerline_points.size();}
//            if (min_ind_prev<0){ min_ind_prev += centerline_points.size();}
//
//            //closest line segment: either [min_ind ,min_ind+1] or [min_ind,min_ind-1]
//            if (pow(x-centerline_points.at(min_ind_next).x, 2) + pow(y-centerline_points.at(min_ind_next).y, 2) <
//                    pow(x-centerline_points.at(min_ind_prev).x, 2) + pow(y-centerline_points.at(min_ind_prev).y, 2)){
//                    second_min_ind = min_ind_next;
//            }
//            else{
//                second_min_ind = min_ind_prev;
//            }
//
//            p(0) = x;  p(1) = y;
//            p0(0) = centerline_points.at(min_ind).x;  p0(1) = centerline_points.at(min_ind).y;
//            p1(0) = centerline_points.at(second_min_ind).x;  p1(1) = centerline_points.at(second_min_ind).y;
//
//            double projection = abs((p - p0).dot(p1 - p0)/(p1 - p0).norm());
//            double theta;
//
//            if (min_ind > second_min_ind){
//                theta = centerline_points.at(min_ind).theta - projection;
//            }
//            else {
//                if (min_ind == 0 && second_min_ind == centerline_points.size()-1) {
//                    theta = length - projection;
//                } else {
//                    theta = centerline_points.at(min_ind).theta + projection;
//                }
//            }
//            if (theta>length){ theta -= length;}
//            if (theta<0){ theta += length;}
            int min_ind;
            double min_dist2 = 10000000.0;
            for (int i=0; i<centerline_points.size(); i++){
                double dist2 = pow(x-centerline_points.at(i).x, 2) + pow(y-centerline_points.at(i).y, 2);
                if( dist2 < min_dist2){
                    min_dist2 = dist2;
                    min_ind = i;
                }
            }
            return min_ind*space;
    }

    void wrapTheta(double& theta){
        while(theta > length){ theta -= length;}
        while(theta < 0){theta += length; }
    }

    double x_eval(double theta){
        wrapTheta(theta);
        return X_spline(theta);
    }

    double y_eval(double theta){
        wrapTheta(theta);
        return Y_spline(theta);
    }

    double x_eval_d(double theta){
        wrapTheta(theta);
        return X_spline.eval_d(theta);
    }

    double y_eval_d(double theta){
        wrapTheta(theta);
        return Y_spline.eval_d(theta);
    }

    double x_eval_dd(double theta){
        wrapTheta(theta);
        return X_spline.eval_dd(theta);
    }

    double y_eval_dd(double theta){
        wrapTheta(theta);
        return Y_spline.eval_dd(theta);
    }

    double getPhi(double theta){
        wrapTheta(theta);

        double dx_dtheta = X_spline.eval_d(theta);
        double dy_dtheta = Y_spline.eval_d(theta);

        return atan2(dy_dtheta, dx_dtheta);
    }

    double getLeftHalfWidth(double theta){
       // wrapTheta(theta);
        int ind = static_cast<int>(floor(theta/space));
        ind = max(0, min(int(centerline_points.size()-1), ind));
        return centerline_points.at(ind).left_half_width;
    }

    double getRightHalfWidth(double theta){
        // wrapTheta(theta);
        int ind = static_cast<int>(floor(theta/space));
        ind = max(0, min(int(centerline_points.size()-1), ind));
        return centerline_points.at(ind).right_half_width;
    }

    void setHalfWidth(double theta, double left_val, double right_val){
        // wrapTheta(theta);
        int ind = static_cast<int>(floor(theta/space));
        ind = max(0, min(int(centerline_points.size()-1), ind));
        centerline_points.at(ind).left_half_width = left_val;
        centerline_points.at(ind).right_half_width = right_val;
    }

    double getcenterline_pointsCurvature(double theta){
        return (X_spline.eval_d(theta)*Y_spline.eval_dd(theta) - Y_spline.eval_d(theta)*X_spline.eval_dd(theta))/
                pow((pow(X_spline.eval_d(theta),2) + pow(Y_spline.eval_d(theta),2)), 1.5);
    }

    double getcenterline_pointsRadius(double theta){
        return 1.0/(getcenterline_pointsCurvature(theta));
    }
};