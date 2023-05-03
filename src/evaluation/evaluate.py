import numpy as np
import torch
from evaluation.utils.saving import save_stats_txt, save_best_stats_txt, check_gpu
from evaluation.utils.similarity_measures import get_RMSE, get_FD, get_DTWD
from agent.utils.dynamical_system_operations import denormalize_state
from scipy.spatial.transform import Rotation


class Evaluate():
    """
    Class for evaluating learned dynamical system
    """
    def __init__(self, learner, data, params, verbose=True):
        self.learner = learner
        self.verbose = verbose

        # Params file parameters
        self.fixed_point_iteration_thr = params.fixed_point_iteration_thr
        self.dim_space = params.manifold_dimensions
        if params.space == 'sphere' or params.space == 'euclidean_sphere':
            self.dim_space += 1
        self.dim_state = self.dim_space * params.dynamical_system_order
        self.dim_manifold = params.manifold_dimensions
        self.ignore_n_spurious = params.ignore_n_spurious
        self.multi_motion = params.multi_motion
        self.quanti_eval = params.quanti_eval
        self.quali_eval = params.quali_eval
        self.diffeo_quanti_eval = params.diffeo_quanti_eval
        self.diffeo_quali_eval = params.diffeo_quali_eval
        self.diffeo_eval = self.diffeo_quanti_eval or self.diffeo_quali_eval
        self.density = params.density
        self.simulated_trajectory_length = params.simulated_trajectory_length
        self.dynamical_system_order = params.dynamical_system_order
        self.space = params.space
        self.skip_stability_evaluation = params.stabilization_loss_weight == 0  # for behavioral cloning

        # Parameters data processor
        self.primitive_ids = np.array(data['demonstrations primitive id'])
        self.n_primitives = data['n primitives']
        self.delta_t_eval = data['delta t eval']
        self.n_trajectories = data['n demonstrations']
        self.demonstrations_train = data['demonstrations train']
        self.eval_indexes = data['eval indexes']
        self.max_trajectory_length = data['max demonstration length']
        self.goals = data['goals training']
        self.demonstrations_eval = data['demonstrations raw']
        self.x_min = np.array(data['x min'])
        self.x_max = np.array(data['x max'])
        if self.space == 'sphere':
            self.radius = data['radius']

        # Variables
        self.best_metric, self.best_n_spurious = 1e7, 1e7
        self.best_RMSE, self.best_DTWD, self.best_FD = 0, 0, 0
        self.RMSE, self.DTWD, self.FD = [], [], []
        self.n_spurious, self.mean_distance_to_goal = [], []
        self.best_model = False

    def get_initial_states_grid(self):
        """
        Samples initial states from a grid in the state space
        """
        # If data in sphere, map sampled points to cartesian coordinates
        if self.space == 'sphere':
            if self.dim_manifold == 2:
                theta, phi = np.linspace(0, np.pi, self.density), np.linspace(0, 2 * np.pi, self.density)
                THETA, PHI = np.meshgrid(theta, phi)
                grid_x = self.radius * np.sin(THETA) * np.cos(PHI)
                grid_y = self.radius * np.sin(THETA) * np.sin(PHI)
                grid_z = self.radius * np.cos(THETA)
                grid = [grid_x, grid_y, grid_z]

                # points_sphere = np.random.uniform(low=-1, high=1, size=(self.dim_manifold, self.density**self.dim_manifold)) * np.pi
                # grid_x = self.radius * np.sin(points_sphere[0]) * np.cos(points_sphere[1])
                # grid_y = self.radius * np.sin(points_sphere[0]) * np.sin(points_sphere[1])
                # grid_z = self.radius * np.cos(points_sphere[0])
                # grid = [grid_x, grid_y, grid_z]
            elif self.dim_manifold == 3:
                points_sphere = np.random.uniform(low=-1, high=1, size=(self.density ** self.dim_manifold, self.dim_manifold)) * np.pi
                rot = Rotation.from_euler('xyz', points_sphere)
                grid = rot.as_quat().T
            else:
                raise NameError('Dimension manifold too large, not implemented.')
        elif self.space == 'euclidean_sphere':
            points_sphere = np.random.uniform(low=-1, high=1, size=(self.density ** self.dim_manifold, self.dim_manifold))
            rot = Rotation.from_euler('xyz', points_sphere[:, 3:] * np.pi)
            quat = rot.as_quat()
            grid = np.concatenate([points_sphere[:, :3], quat], axis=1).T
        else:
            # Create workspace grid [-1, 1] x [-1, 1] x ...
            starting_points = np.linspace(-1, 1, self.density)

            starting_points = [starting_points] * self.dim_manifold  # repeat points per dimension workspace
            grid = np.meshgrid(*starting_points)

        # Transform grid into tensor that pytorch can use
        initial_positions_grid = torch.empty(0)
        for i in range(self.dim_space):
            initial_positions_grid = torch.cat([initial_positions_grid,
                                                torch.from_numpy(grid[i].reshape(-1, 1)).float()], dim=1)

        initial_positions_grid = initial_positions_grid.cuda()

        # Get initial derivatives and append to initial states (for second order systems)
        initial_derivatives_grid = torch.zeros([initial_positions_grid.shape[0], self.dim_state - self.dim_space]).cuda()

        # Get initial states
        initial_states_grid = torch.cat([initial_positions_grid, initial_derivatives_grid], dim=1)

        return initial_states_grid, grid

    def get_initial_states_demos(self, primitive_id):
        """
        Gets initial states present in the demonstrations
        """
        demos = self.demonstrations_train[self.primitive_ids == primitive_id]

        # Get initial positions
        initial_positions_demos = torch.empty(0)
        for i in range(self.dim_space):
            initial_positions_demos = torch.cat([initial_positions_demos,
                                                 torch.from_numpy(demos[:, 0, i, 0].reshape(-1, 1)).float()], dim=1)

        initial_positions_demos = initial_positions_demos.cuda()

        # Get initial derivatives and append to initial states (second order systems)
        initial_derivatives_demos = torch.zeros([initial_positions_demos.shape[0], self.dim_state - self.dim_space]).cuda()

        # Get initial states
        initial_states_demos = torch.cat([initial_positions_demos, initial_derivatives_demos], dim=1)

        return initial_states_demos

    def simulate_system(self, primitive_id, space='task', **kwargs):
        """
        Simulates dynamical system when starting from grid initial states and demonstrations initial states
        """
        # Get initial states of demonstrations
        initial_states_demos = self.get_initial_states_demos(primitive_id)

        # Get equally-spaced initial states through the workspace
        initial_states_grid, grid = self.get_initial_states_grid()

        # Get n demos in primitive
        n_trajectories_primitive = self.demonstrations_train[self.primitive_ids == primitive_id].shape[0]

        # Get primitive number to feed model
        primitive_type = torch.ones(self.density ** self.dim_manifold + n_trajectories_primitive).cuda() * primitive_id

        # Combine states
        initial_states = torch.cat([initial_states_demos, initial_states_grid], dim=0)

        # Simulate trajectories
        dynamical_system = self.learner.init_dynamical_system(initial_states, primitive_type)
        visited_states = dynamical_system.simulate(self.simulated_trajectory_length, space=space, **kwargs)

        # Separate grid/demos states
        visited_states_demos = visited_states[:, :n_trajectories_primitive]
        visited_states_grid = visited_states[:, n_trajectories_primitive:]

        # Out dict
        results = {'visited states demos': visited_states_demos,
                   'visited states grid': visited_states_grid,
                   'initial states grid': initial_states_grid,
                   'grid': grid}

        return results

    def get_vector_field(self, initial_states, primitive_id, **kwargs):
        """
        Computes velocity of initial states
        """
        # Get primitive number to feed model
        primitive_type = torch.ones(self.density ** self.dim_manifold).cuda() * primitive_id

        # Compute evaluation delta t
        delta_t_eval = np.mean(self.delta_t_eval)

        # Do one transition
        with torch.no_grad():
            dynamical_system = self.learner.init_dynamical_system(initial_states, primitive_type)
            x_t = dynamical_system.transition(space='task', **kwargs)['desired state']

        # Denormalize states
        x_init_denorm = denormalize_state(initial_states[:, :self.dim_state].cpu().detach().numpy(),
                                          x_min=np.array(self.x_min),
                                          x_max=np.array(self.x_max))

        x_t_denorm = denormalize_state(x_t[:, :self.dim_state].cpu().detach().numpy(),
                                       x_min=np.array(self.x_min),
                                       x_max=np.array(self.x_max))

        # Compute differences in X
        vel = (x_t_denorm - x_init_denorm) / delta_t_eval

        return vel.reshape(self.density, self.density, -1)

    def get_stability_metrics(self, attractor, goal):
        """
        Based on trajectories obtained when starting from grid initial states, computes stability metrics
        """
        # Get goal
        goal = denormalize_state(np.array(goal), self.x_min, self.x_max)

        # Compute distance between goal and last point
        distance_to_goal = np.linalg.norm(attractor - goal, axis=1)

        # Find spurious/unsuc. trajectory and mean distance to goal
        n_spurious = np.sum(distance_to_goal > self.fixed_point_iteration_thr)
        mean_dist_to_goal = np.mean(distance_to_goal)

        # Append results to history
        self.mean_distance_to_goal.append(mean_dist_to_goal)
        self.n_spurious.append(n_spurious)

        # Get results
        metrics_stab = {'n spurious': n_spurious,
                        'mean dist to goal': mean_dist_to_goal}

        return metrics_stab

    def get_accuracy_metrics(self, visited_states, demonstrations_eval, max_trajectory_length, eval_indexes):
        """
        Gets accuracy metrics to evaluate the performance of the learned dynamical system
        """

        # Denormalize and preprocess trajectories
        sim_trajectories = denormalize_state(visited_states[:, :, :self.dim_space], self.x_min, self.x_max)
        demos = self.preprocess_demonstrations_eval(demonstrations_eval, visited_states.shape[1],
                                                    max_trajectory_length)

        # Calculate RMSE
        RMSE = get_RMSE(sim_trajectories, demos, eval_indexes, verbose=self.verbose)

        # Calculate DTWD
        DTWD = get_DTWD(sim_trajectories, demos, eval_indexes, verbose=self.verbose)

        # Calculate Frechet Distance
        FD = get_FD(sim_trajectories, demos, eval_indexes, verbose=self.verbose)

        # Get average over demonstrations
        mean_RMSE = np.array(RMSE).mean()
        mean_DTWD = np.array(DTWD).mean()
        mean_FD = np.array(FD).mean()

        # Sum metrics
        self.metrics_sum = mean_RMSE + mean_DTWD + mean_FD

        # Append results to history
        self.RMSE.append(RMSE)
        self.DTWD.append(DTWD)
        self.FD.append(FD)

        # Get results
        results = {'RMSE': mean_RMSE,
                   'DTWD': mean_DTWD,
                   'FD': mean_FD,
                   'metrics sum': self.metrics_sum}

        return results

    def preprocess_demonstrations_eval(self, demonstrations_eval, n_trajectories, max_trajectory_length):
        """
        Add zeros at the end of demonstrations so that they have the same length and can be stored in a numpy array
        """

        # Initialize padded demos
        demos_padded = np.empty([max_trajectory_length, n_trajectories, self.dim_space])

        # Add zeros to each trajectory such that they all have the same length as the longest one
        for i in range(n_trajectories):
            for j in range(self.dim_space):
                demonstrations_eval_i_j = demonstrations_eval[i][j]
                length_diff = max_trajectory_length - len(demonstrations_eval_i_j)
                demonstrations_eval_i_j = np.append(demonstrations_eval_i_j, np.zeros(length_diff))
                demos_padded[:, i, j] = demonstrations_eval_i_j

        return demos_padded
    
    def compute_quanti_eval(self, sim_results, attractor, primitive_id):
        """
        Computes quantitative evaluation metrics
        """

        # Ugly fix for evaluating in multi-models TODO: fix
        demos_eval = []
        eval_index = []
        for i in range(len(self.primitive_ids)):
            if self.primitive_ids[i] == 0:
                demos_eval.append(self.demonstrations_eval[i])
                eval_index.append(self.eval_indexes[i])

        # Get accuracy metrics
        # metrics_acc = self.get_accuracy_metrics(sim_results['visited states demos'],
        #                                         self.demonstrations_eval,
        #                                         self.max_trajectory_length,
        #                                         self.eval_indexes)
        metrics_acc = self.get_accuracy_metrics(sim_results['visited states demos'],
                                                demos_eval,
                                                self.max_trajectory_length,
                                                eval_index)
        # Get stability metrics
        metrics_stab = self.get_stability_metrics(attractor, self.goals[primitive_id])

        # Select best model
        self.select_best_model(metrics_acc, metrics_stab)
        
        return metrics_acc, metrics_stab

    def select_best_model(self, metrics_acc, metrics_stab):
        """
        Based on the computed metrics, selects best model so far
        """
        self.best_model = False

        # Select best model
        lower_metric_error = metrics_acc['metrics sum'] < self.best_metric

        if not self.ignore_n_spurious:
            fewer_spurious = metrics_stab['n spurious'] < self.best_n_spurious
            same_spurious = metrics_stab['n spurious'] == self.best_n_spurious
        else:  # we only pass if the metric error is lower
            fewer_spurious = False  # ignore this value in the logical statement
            same_spurious = True  # set this one to true, so that we depend on the metric error

        if fewer_spurious or (same_spurious and lower_metric_error) or (self.skip_stability_evaluation and lower_metric_error):  # give priority to number of spurious attractors
            self.best_n_spurious = metrics_stab['n spurious']
            self.best_metric = metrics_acc['metrics sum']
            self.best_RMSE = metrics_acc['RMSE']
            self.best_DTWD = metrics_acc['DTWD']
            self.best_FD = metrics_acc['FD']
            self.best_model = True

    def compute_quali_eval(self, sim_results, attractor, primitive_id, iteration, save_path):
        """
        Qualitative evaluation, to be computed in subclass
        """
        raise NotImplementedError('Should be implemented in subclass!')

    def run(self, iteration, save_path=None):
        """
        Runs complete evaluation
        """
        metrics_acc, metrics_stab = {}, {}

        # Iterate when multi-motion learning
        for primitive_id in range(self.n_primitives):
            # Simulate trajectories starting from initial conditions demos and from equally-spaced grid
            sim_results = self.simulate_system(primitive_id)

            # Get last point trajectories grid
            attractor = denormalize_state(sim_results['visited states grid'][-1, :, :self.dim_space], self.x_min, self.x_max)

            if self.quanti_eval:
                if primitive_id == 0:  # TODO: ugly fix for multiple models, fix!
                    metrics_acc, metrics_stab = self.compute_quanti_eval(sim_results, attractor, primitive_id)

            if self.quali_eval:
                self.compute_quali_eval(sim_results, attractor, primitive_id, iteration, save_path)

        return metrics_acc, metrics_stab

    def save_progress(self, save_path, i, model, writer):
        """
        Save and log
        """

        # Log metrics to tensorboard
        writer.add_scalar('eval/RMSE', np.mean(self.RMSE[-1]), i)
        writer.add_scalar('eval/DTWD', np.mean(self.DTWD[-1]), i)
        writer.add_scalar('eval/FD', np.mean(self.FD[-1]), i)
        writer.add_scalar('eval/n_spurious', self.n_spurious[-1], i)
        writer.add_scalar('eval/mean_distance_to_goal', self.mean_distance_to_goal[-1], i)

        # Save metrics history
        np.save(save_path + 'stats/n_unsuccesful_trajs', self.n_spurious)
        np.save(save_path + 'stats/mean_distance_to_goal', self.mean_distance_to_goal)
        np.save(save_path + 'stats/RMSE', self.RMSE)
        np.save(save_path + 'stats/DTWD', self.DTWD)
        np.save(save_path + 'stats/FD', self.FD)

        # Save states
        save_stats_txt(save_path, self.n_spurious, self.metrics_sum, self.RMSE, self.DTWD, self.FD, i)

        if self.best_model or self.multi_motion:  # metric evaluation is not yet properly done in multi task case
            # Save torch
            torch.save(model.state_dict(), save_path + 'model')

            # Save best stats in txt
            gpu_status = check_gpu()
            save_best_stats_txt(save_path, self.best_n_spurious, self.best_metric, self.best_RMSE, self.best_DTWD,
                                self.best_FD, gpu_status, i)
