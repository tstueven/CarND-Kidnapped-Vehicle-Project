/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
    num_particles = 100;  // TODO: Set the number of particles_

    double std_x = std[0];
    double std_y = std[1];
    double std_theta = std[2];

    std::normal_distribution<double> dist_x(x, std_x);
    std::normal_distribution<double> dist_y(y, std_y);
    std::normal_distribution<double> dist_theta(theta, std_theta);

    for (int i = 0; i < 100; ++i)
    {
        Particle p;
        p.x = dist_x(gen_);
        p.y = dist_y(gen_);
        p.theta = angle_to_interval(dist_theta(gen_));
        p.id = i;
        p.weight = 1;
        particles_.push_back(std::move(p));
        weights.push_back(1);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate)
{
    double std_x = std_pos[0];
    double std_y = std_pos[1];
    double std_theta = std_pos[2];

    std::normal_distribution<double> dist_x(0, std_x);
    std::normal_distribution<double> dist_y(0, std_y);
    std::normal_distribution<double> dist_theta(0, std_theta);

    for (auto &p : this->particles_)
    {
        if (std::abs(yaw_rate) < 1e-4)
        {
            p.x += velocity * delta_t * std::cos(p.theta);
            p.y += velocity * delta_t * std::sin(p.theta);
        }
        else
        {
            p.x += (velocity / yaw_rate) *
                   (std::sin(p.theta + yaw_rate * delta_t) - std::sin(p.theta));
            p.y += (velocity / yaw_rate) *
                   (std::cos(p.theta) - std::cos(p.theta + yaw_rate * delta_t));
        }
        p.theta += yaw_rate * delta_t;

        p.x += dist_x(gen_);
        p.y += dist_y(gen_);
        p.theta += angle_to_interval(dist_theta(gen_));
    }
}

void ParticleFilter::dataAssociation(const vector<LandmarkObs> &predicted,
                                     vector<LandmarkObs> &observations)
{
    for (auto &observation : observations)
    {
        double min_dist_sq = std::numeric_limits<double>::max();
        for (const auto &pred : predicted)
        {
            double distance_sq = observation.dist_sq(pred);
            if (distance_sq < min_dist_sq)
            {
                min_dist_sq = distance_sq;
                observation.dist_to_nearest_x = observation.x - pred.x;
                observation.dist_to_nearest_y = observation.y - pred.y;
                observation.id = pred.id;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks)
{
    /**
     * TODO: Update the weights of each particle using a mult-variate Gaussian
     *   distribution. You can read more about this distribution here:
     *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
     * NOTE: The observations are given in the VEHICLE'S coordinate system.
     *   Your particles_ are located according to the MAP'S coordinate system.
     *   You will need to transform between the two systems. Keep in mind that
     *   this transformation requires both rotation AND translation (but no scaling).
     *   The following is a good resource for the theory:
     *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
     *   and the following is a good resource for the actual equation to implement
     *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
     */
    const double sensor_range_sq = sensor_range * sensor_range;

    for (auto &p : particles_)
    {
        vector<LandmarkObs> landmarks_in_range{};
        for (const auto &landmark : map_landmarks.landmark_list)
        {
            if (dist_sq(p.x, p.y, landmark.x_f, landmark.y_f) <= sensor_range_sq)
            {
                landmarks_in_range.emplace_back(landmark.id_i, landmark.x_f, landmark.y_f);
            }
        }

//        vector<LandmarkObs> global_observations{};
//        global_observations.reserve(observations.size());
//        for (const auto &local_obs : observations)
//        {
//            global_observations.emplace_back(p.obervation_to_global(local_obs));
//        }
//        dataAssociation(landmarks_in_range, global_observations);

        // I chose this way since I think the landmark measurement uncertainty has to be dealt with
        // in particle coordinates.
        vector<LandmarkObs> local_landmarks{};
        local_landmarks.reserve(landmarks_in_range.size());
        for (const auto &landmark : landmarks_in_range)
        {
            local_landmarks.emplace_back(p.landmark_to_local(landmark));
        }
        auto observations_copy_to_assign_landmarks = observations;
        dataAssociation(local_landmarks, observations_copy_to_assign_landmarks);

        p.weight = 1;
        for (const auto &obs : observations_copy_to_assign_landmarks)
        {
            double weight = gaussian(obs.dist_to_nearest_x, obs.dist_to_nearest_y, std_landmark[0],
                                     std_landmark[1]);
            weight = weight > 1e-8 ? weight : 1e-8;
            p.weight *= weight;
        }
    }
}

void ParticleFilter::resample()
{
    /**
     * TODO: Resample particles_ with replacement with probability proportional
     *   to their weight.
     * NOTE: You may find std::discrete_distribution helpful here.
     *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
     */
     

}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y)
{
    // particle: the particle to which assign each listed association,
    //   and association's (x,y) world coordinates mapping
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(const Particle &best)
{
    vector<int> v = best.associations;
    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseCoord(const Particle &best, const string &coord)
{
    vector<double> v;

    if (coord == "X")
    {
        v = best.sense_x;
    }
    else
    {
        v = best.sense_y;
    }

    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}