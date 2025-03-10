/**
 * particle_filter.h
 * 2D particle filter class.
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include <string>
#include <vector>
#include <random>
#include "helper_functions.h"

struct Particle
{
    int id;
    double x;
    double y;
    double theta;
    double weight;
    std::vector<int> associations;
    std::vector<double> sense_x;
    std::vector<double> sense_y;

    LandmarkObs obervation_to_global(const LandmarkObs &obs)
    {
        LandmarkObs out = obs;
        auto[rot_x, rot_y] = rotate_local_to_global(obs.x, obs.y, this->theta);
        out.x = this->x + rot_x;
        out.y = this->y + rot_y;
        return out;
    }

    LandmarkObs landmark_to_local(const LandmarkObs &obs)
    {
        LandmarkObs out = obs;
        auto[rot_x, rot_y] = rotate_global_to_local(obs.x - this->x, obs.y - this->y,
                                                    this->theta);
        out.x = rot_x;
        out.y = rot_y;
        return out;
    }

};


class ParticleFilter
{
public:
    // Constructor
    // @param num_particles Number of particles_
    ParticleFilter() : num_particles(0), is_initialized(false) {}

    // Destructor
    ~ParticleFilter() = default;

    /**
     * init Initializes particle filter by initializing particles_ to Gaussian
     *   distribution around first position and all the weights_ to 1.
     * @param x Initial x position [m] (simulated estimate from GPS)
     * @param y Initial y position [m]
     * @param theta Initial orientation [rad]
     * @param std[] Array of dimension 3 [standard deviation of x [m],
     *   standard deviation of y [m], standard deviation of yaw [rad]]
     */
    void init(double x, double y, double theta, double std[]);

    /**
     * prediction Predicts the state for the next time step
     *   using the process model.
     * @param delta_t Time between time step t and t+1 in measurements [s]
     * @param std_pos[] Array of dimension 3 [standard deviation of x [m],
     *   standard deviation of y [m], standard deviation of yaw [rad]]
     * @param velocity Velocity of car from t to t+1 [m/s]
     * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
     */
    void prediction(double delta_t, double std_pos[], double velocity,
                    double yaw_rate);

    /**
     * dataAssociation Finds which observations correspond to which landmarks
     *   (likely by using a nearest-neighbors data association).
     * @param predicted Vector of predicted landmark observations
     * @param observations Vector of landmark observations
     */
    void dataAssociation(const std::vector<LandmarkObs> &predicted,
                         std::vector<LandmarkObs> &observations);

    /**
     * updateWeights Updates the weights_ for each particle based on the likelihood
     *   of the observed measurements.
     * @param sensor_range Range [m] of sensor
     * @param std_landmark[] Array of dimension 2
     *   [Landmark measurement uncertainty [x [m], y [m]]]
     * @param observations Vector of landmark observations
     * @param map Map class containing map landmarks
     */
    void updateWeights(double sensor_range, double std_landmark[],
                       const std::vector<LandmarkObs> &observations,
                       const Map &map_landmarks);

    /**
     * resample Resamples from the updated set of particles_ to form
     *   the new set of particles_.
     */
    void resample();

    /**
     * Set a particles_ list of associations, along with the associations'
     *   calculated world x,y coordinates
     * This can be a very useful debugging tool to make sure transformations
     *   are correct and assocations correctly connected
     */
    void SetAssociations(Particle &particle, const std::vector<int> &associations,
                         const std::vector<double> &sense_x,
                         const std::vector<double> &sense_y);

    /**
     * initialized Returns whether particle filter is initialized yet or not.
     */
    const bool initialized() const
    {
        return is_initialized;
    }

    /**
     * Used for obtaining debugging information related to particles_.
     */
    std::string getAssociations(const Particle &best);

    std::string getSenseCoord(const Particle &best, const std::string &coord);

    // Set of current particles_
    std::vector<Particle> particles_;

private:
    // Number of particles_ to draw
    int num_particles;

    // Flag, if filter is initialized
    bool is_initialized;

    // Vector of weights_ of all particles_
    std::vector<double> weights_;

    std::default_random_engine gen_{};
};

#endif  // PARTICLE_FILTER_H_