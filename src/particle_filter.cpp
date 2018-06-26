/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
static int N_P = 100;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  // create normal distributions for x, y, and theta
  std::normal_distribution<double> dis_x(x, std[0]);
  std::normal_distribution<double> dis_y(y, std[1]);
  std::normal_distribution<double> dis_theta(theta, std[2]);

  std::default_random_engine gen;

  // resize the vectors of particles and weights
  num_par = N_P;
  particles.resize(num_par);
  //weights.resize(num_particles);

  // generate the particles
  for(auto& p: particles){
    p.x = dis_x(gen);
    p.y = dis_y(gen);
    p.theta = dis_theta(gen);
    p.weight = 1;
  }

  is_initialized = true;


}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  std::default_random_engine gen;

  // generate random Gaussian noise
  std::normal_distribution<double> N_x(0, std_pos[0]);
  std::normal_distribution<double> N_y(0, std_pos[1]);
  std::normal_distribution<double> N_theta(0, std_pos[2]);

  for(auto& p: particles){

    // add measurements to each particle
    if( fabs(yaw_rate) < 0.0001){  // constant velocity
      p.x += velocity * delta_t * cos(p.theta);
      p.y += velocity * delta_t * sin(p.theta);

    } else{
      p.x += velocity / yaw_rate * ( sin( p.theta + yaw_rate*delta_t ) - sin(p.theta) );
      p.y += velocity / yaw_rate * ( cos( p.theta ) - cos( p.theta + yaw_rate*delta_t ) );
      p.theta += yaw_rate * delta_t;
    }

    // predicted particles with added sensor noise
    p.x += N_x(gen);
    p.y += N_y(gen);
    p.theta += N_theta(gen);
  }

}




void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
  for(auto& ob: observations){
    double miD = std::numeric_limits<float>::max();

    for(const auto& pre: predicted){
      double dista = dist(ob.x, ob.y, pre.x, pre.y);
      if( miD > dista){
        miD = dista;
        ob.id = pre.id;
      }
    }
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  for(auto& p: particles){
    p.weight = 1.0;

    //  collect valid landmarks
    vector<LandmarkObs> prediction;
    for(const auto& lm: map_landmarks.landmark_list){
      double dista = dist(p.x, p.y, lm.x_f, lm.y_f);
      if( dista < sensor_range){ // if the landmark is within the sensor range, save it to predictions
        prediction.push_back(LandmarkObs{lm.id_i, lm.x_f, lm.y_f});
      }
    }

    //  convert observations coordinates from vehicle to map
    vector<LandmarkObs> obs_map;
    double cos_theta = cos(p.theta);
    double sin_theta = sin(p.theta);

    for(const auto& obs: observations){
      LandmarkObs tp;
      tp.x = obs.x * cos_theta - obs.y * sin_theta + p.x;
      tp.y = obs.x * sin_theta + obs.y * cos_theta + p.y;
      //tmp.id = obs.id; // maybe an unnecessary step, since the each obersation will get the id from dataAssociation step.
      obs_map.push_back(tp);
    }

    //  find landmark index for each observation
    dataAssociation(prediction, obs_map);

    //  compute the particle's weight:
    // see equation this link:
    for(const auto& obs_m: obs_map){

      Map::single_landmark_s landmark = map_landmarks.landmark_list.at(obs_m.id-1);
      double x_term = pow(obs_m.x - landmark.x_f, 2) / (2 * pow(std_landmark[0], 2));
      double y_term = pow(obs_m.y - landmark.y_f, 2) / (2 * pow(std_landmark[1], 2));
      double w = exp(-(x_term + y_term)) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
      p.weight *=  w;
    }

    weights.push_back(p.weight);

  }

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  // generate distribution according to weights
 std::random_device r;
 std::mt19937 gen(r());
 std::discrete_distribution<> dist(weights.begin(), weights.end());

 // create resampled particles
 vector<Particle> re_par;
 re_par.resize(num_par);

 // resample the particles according to weights
 for(int i=0; i<num_par; i++){
   int idx = dist(gen);
   re_par[i] = particles[idx];
 }

 // assign the resampled_particles to the previous particles
 particles = re_par;

 // clear the weight vector for the next round
 weights.clear();

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations.clear();
  	particle.sense_x.clear();
  	particle.sense_y.clear();

  	particle.associations= associations;
   	particle.sense_x = sense_x;
   	particle.sense_y = sense_y;

   	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

