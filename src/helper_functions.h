/**
 * helper_functions.h
 * Some helper functions for the 2D particle filter.
 *
 * Created on: Dec 13, 2016
 * Author: Tiffany Huang
 */

#ifndef HELPER_FUNCTIONS_H_
#define HELPER_FUNCTIONS_H_

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "map.h"

// for portability of M_PI (Vis Studio, MinGW, etc.)
#ifndef M_PI
const double M_PI = 3.14159265358979323846;
#endif

/**
 * Struct representing one position/control measurement.
 */
struct control_s
{
    double velocity;  // Velocity [m/s]
    double yawrate;   // Yaw rate [rad/s]
};

/**
 * Struct representing one ground truth position.
 */
struct ground_truth
{
    double x;     // Global vehicle x position [m]
    double y;     // Global vehicle y position
    double theta; // Global vehicle yaw [rad]
};

/**
 * Computes the squared Euclidean distance between two 2D points.
 * @param (x1,y1) x and y coordinates of first point
 * @param (x2,y2) x and y coordinates of second point
 * @output squared Euclidean distance between two 2D points
 */
inline double dist_sq(double x1, double y1, double x2, double y2)
{
    return (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
}

/**
 * Struct representing one landmark observation measurement.
 */
struct LandmarkObs
{
    int id;     // Id of matching landmark in the map.
    double x;   // Local (vehicle coords) x position of landmark observation [m]
    double y;   // Local (vehicle coords) y position of landmark observation [m]
    double dist_to_nearest_x; // x dist to closest landmark
    double dist_to_nearest_y; // y dist to closest landmark

    LandmarkObs() = default;

    LandmarkObs(int id, double x, double y) : id{id}, x{x}, y{y} {}

    double dist_sq(const LandmarkObs &other) const
    {
        return ::dist_sq(x, y, other.x, other.y);
    }
};

/**
 * Computes the Euclidean distance between two 2D points.
 * @param (x1,y1) x and y coordinates of first point
 * @param (x2,y2) x and y coordinates of second point
 * @output Euclidean distance between two 2D points
 */
inline double dist(double x1, double y1, double x2, double y2)
{
    return sqrt(dist_sq(x1, y1, x2, y2));
}

/**
 * Computes the error between ground truth and particle filter data.
 * @param (gt_x, gt_y, gt_theta) x, y and theta of ground truth
 * @param (pf_x, pf_y, pf_theta) x, y and theta of particle filter
 * @output Error between ground truth and particle filter data.
 */
inline double *getError(double gt_x, double gt_y, double gt_theta, double pf_x,
                        double pf_y, double pf_theta)
{
    static double error[3];
    error[0] = fabs(pf_x - gt_x);
    error[1] = fabs(pf_y - gt_y);
    error[2] = fabs(pf_theta - gt_theta);
    error[2] = fmod(error[2], 2.0 * M_PI);
    if (error[2] > M_PI)
    {
        error[2] = 2.0 * M_PI - error[2];
    }
    return error;
}

/**
 * Reads map data from a file.
 * @param filename Name of file containing map data.
 * @output True if opening and reading file was successful
 */
inline bool read_map_data(std::string filename, Map &map)
{
    // Get file of map
    std::ifstream in_file_map(filename.c_str(), std::ifstream::in);
    // Return if we can't open the file
    if (!in_file_map)
    {
        return false;
    }

    // Declare single line of map file
    std::string line_map;

    // Run over each single line
    while (getline(in_file_map, line_map))
    {

        std::istringstream iss_map(line_map);

        // Declare landmark values and ID
        float landmark_x_f, landmark_y_f;
        int id_i;

        // Read data from current line to values
        iss_map >> landmark_x_f;
        iss_map >> landmark_y_f;
        iss_map >> id_i;

        // Declare single_landmark
        Map::single_landmark_s single_landmark_temp;

        // Set values
        single_landmark_temp.id_i = id_i;
        single_landmark_temp.x_f = landmark_x_f;
        single_landmark_temp.y_f = landmark_y_f;

        // Add to landmark list of map
        map.landmark_list.push_back(single_landmark_temp);
    }
    return true;
}

/**
 * Reads control data from a file.
 * @param filename Name of file containing control measurements.
 * @output True if opening and reading file was successful
 */
inline bool read_control_data(std::string filename,
                              std::vector<control_s> &position_meas)
{
    // Get file of position measurements
    std::ifstream in_file_pos(filename.c_str(), std::ifstream::in);
    // Return if we can't open the file
    if (!in_file_pos)
    {
        return false;
    }

    // Declare single line of position measurement file:
    std::string line_pos;

    // Run over each single line:
    while (getline(in_file_pos, line_pos))
    {

        std::istringstream iss_pos(line_pos);

        // Declare position values:
        double velocity, yawrate;

        // Declare single control measurement:
        control_s meas;

        //read data from line to values:
        iss_pos >> velocity;
        iss_pos >> yawrate;

        // Set values
        meas.velocity = velocity;
        meas.yawrate = yawrate;

        // Add to list of control measurements:
        position_meas.push_back(meas);
    }
    return true;
}

/**
 * Reads ground truth data from a file.
 * @param filename Name of file containing ground truth.
 * @output True if opening and reading file was successful
 */
inline bool read_gt_data(std::string filename, std::vector<ground_truth> &gt)
{
    // Get file of position measurements
    std::ifstream in_file_pos(filename.c_str(), std::ifstream::in);
    // Return if we can't open the file
    if (!in_file_pos)
    {
        return false;
    }

    // Declare single line of position measurement file
    std::string line_pos;

    // Run over each single line
    while (getline(in_file_pos, line_pos))
    {

        std::istringstream iss_pos(line_pos);

        // Declare position values
        double x, y, azimuth;

        // Declare single ground truth
        ground_truth single_gt;

        //read data from line to values
        iss_pos >> x;
        iss_pos >> y;
        iss_pos >> azimuth;

        // Set values
        single_gt.x = x;
        single_gt.y = y;
        single_gt.theta = azimuth;

        // Add to list of control measurements and ground truth
        gt.push_back(single_gt);
    }
    return true;
}

/**
 * Reads landmark observation data from a file.
 * @param filename Name of file containing landmark observation measurements.
 * @output True if opening and reading file was successful
 */
inline bool read_landmark_data(const std::string &filename,
                               std::vector<LandmarkObs> &observations)
{
    // Get file of landmark measurements
    std::ifstream in_file_obs(filename.c_str(), std::ifstream::in);
    // Return if we can't open the file
    if (!in_file_obs)
    {
        return false;
    }

    // Declare single line of landmark measurement file
    std::string line_obs;

    // Run over each single line
    while (getline(in_file_obs, line_obs))
    {

        std::istringstream iss_obs(line_obs);

        // Declare position values
        double local_x, local_y;

        //read data from line to values
        iss_obs >> local_x;
        iss_obs >> local_y;

        // Declare single landmark measurement
        LandmarkObs meas;

        // Set values
        meas.x = local_x;
        meas.y = local_y;

        // Add to list of control measurements
        observations.push_back(meas);
    }
    return true;
}

inline double angle_to_interval(double phi)
{
    return atan2(sin(phi), cos(phi));
}

inline std::tuple<double, double> rotate_local_to_global(double x, double y, double theta)
{
    double cos_t = std::cos(theta);
    double sin_t = std::sin(theta);
    double new_x = x * cos_t - y * sin_t;
    double new_y = x * sin_t + y * cos_t;
    return {new_x, new_y};
}

inline std::tuple<double, double> rotate_global_to_local(double x, double y, double theta)
{
    double cos_t = std::cos(theta);
    double sin_t = std::sin(theta);
    double new_x = x * cos_t + y * sin_t;
    double new_y = -x * sin_t + y * cos_t;
    return {new_x, new_y};
}

inline double gaussian(double dist_x, double dist_y, double sigma_x, double sigma_y)
{
    double dist_x_sq = dist_x * dist_x;
    double dist_y_sq = dist_y * dist_y;
    double sigma_x_sq = sigma_x * sigma_x;
    double sigma_y_sq = sigma_y * sigma_y;
    double normalizer = 1. / (2 * M_PI * sigma_x * sigma_y);
    return normalizer *
           std::exp(-1 * (dist_x_sq / (2 * sigma_x_sq) + dist_y_sq / (2 * sigma_y_sq)));
}

#endif  // HELPER_FUNCTIONS_H_
