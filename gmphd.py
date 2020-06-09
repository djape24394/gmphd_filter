import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lin
import time


def multivariate_gaussian(x: np.ndarray, m: np.ndarray, P: np.ndarray):
    """
        Multivatiate Gaussian Distribution
    """
    first_part = 1 / (((2 * np.pi) ** (x.size / 2.0)) * (lin.det(P) ** 0.5))
    second_part = -0.5 * (x - m) @ lin.inv(P) @ (x - m)
    return first_part * np.exp(second_part)


def multivariate_gaussian_with_det_and_inv(x: np.ndarray, m: np.ndarray, detP, invP: np.ndarray):
    """
        Multivariate Gaussian Distribution with provided determinant and inverse of Gaussian mixture
    """
    first_part = 1 / (((2 * np.pi) ** (x.size / 2.0)) * (detP ** 0.5))
    second_part = -0.5 * (x - m) @ invP @ (x - m)
    return first_part * np.exp(second_part)


def clutter_intensity_function(z, lc, surveillance_region):
    '''
    Clutter intensity function, with uniform distribution through the surveillance region, pg. 8
    in "Bayesian Multiple Target Filtering Using Random Finite Sets" by Vo, Vo, Clark.
    :param z:
    :param lc: average number of false detections per time step
    :param surveillance_region: np.ndarray of shape (number_dimensions, 2) giving range(min and max) for each dimension
    '''
    if surveillance_region[0][0] <= z[0] <= surveillance_region[0][1] and surveillance_region[1][0] <= z[1] <= \
            surveillance_region[1][1]:
        # example in two dimensions: lc/((xmax - xmin)*(ymax-ymin))
        return lc / ((surveillance_region[0][1] - surveillance_region[0][0]) * (
                surveillance_region[1][1] - surveillance_region[1][0]))
    else:
        return 0.0


class GaussianMixture:
    def __init__(self, w, m, P):
        """
        The Gaussian mixture class
        inputs:
        - w: list of scalar weights
        - m: list of np.ndarray means
        - m: list of np.ndarray covariance matrices

        Note that constructor creates detP and invP variables which can be used instead of P list for covariance matrix
        determinant and inverse. These lists could be initialized with assign_determinant_and_inverse function
        """
        self.w = w
        self.m = m
        self.P = P
        self.detP = None
        self.invP = None

    def assign_determinant_and_inverse(self, detP, invP):
        self.detP = detP
        self.invP = invP

    def mixture_value(self, x: np.ndarray):
        sum = 0
        if self.detP is None:
            for i in range(len(self.w)):
                sum += self.w[i] * multivariate_gaussian(x, self.m[i], self.P[i])
        else:
            for i in range(len(self.w)):
                sum += self.w[i] * multivariate_gaussian_with_det_and_inv(x, self.m[i], self.detP[i], self.invP[i])
        return sum

    def mixture_component_value_at(self, x: np.ndarray, i: int):
        if self.detP is None:
            return self.w[i] * multivariate_gaussian(x, self.m[i], self.P[i])
        else:
            return self.w[i] * multivariate_gaussian_with_det_and_inv(x, self.m[i], self.detP[i], self.invP[i])

    def mixture_component_values_list(self, x):
        val = []
        if self.detP is None:
            for i in range(len(self.w)):
                val.append(self.w[i] * multivariate_gaussian(x, self.m[i], self.P[i]))
        else:
            for i in range(len(self.w)):
                val.append(self.w[i] * multivariate_gaussian_with_det_and_inv(x, self.m[i], self.detP[i], self.invP[i]))
        return val

    def copy(self):
        w = self.w.copy()
        m = []
        P = []
        for m1 in self.m:
            m.append(m1.copy())
        for P1 in self.P:
            P.append(P1.copy())
        return GaussianMixture(w, m, P)


class GmphdFilter:
    """
        The Gaussian Mixture Probability Hypothesis Density filter implementation. It's based on
        "The Gaussian mixture probability hypothesis density filter" by Vo and Ma.
    """

    def __init__(self, model):
        """
        Note that state x will be np.ndarray. in our model, we assume linear transition and measurement in the
        following form
            x[k] = Fx[k-1] + w[k-1]
            z[k] = Hx[k] + v[k]
        Inputs:
        - model: dictionary which contains the following elements(keys are strings):
               F: state transition matrix
               H: measurement matrix
               Q: process noise covariance matrix(of variable w[k]).
               R: measurement noise covariance matrix(of variable v[k]).
             p_d: probability of target detection
             p_s: probability of target survival

         Spawning model, see paper pg. 5. it's a gaussian mixture conditioned on state
         F_spawn:  d_spawn: Q_spawn: w_spawn: lists of ndarray objects with the same length, see pg. 5

    clutt_int_fun: reference to clutter intensity function, gets only one argument, which is the current measure

               T: U: Jmax: Pruning parameters, see pg. 7.

        birth_GM: The Gaussian Mixture of the birth intensity
        """
        # to do: dtype, copy, improve performance
        self.p_s = model['p_s']
        self.F = model['F']
        self.Q = model['Q']
        self.w_spawn = model['w_spawn']
        self.F_spawn = model['F_spawn']
        self.d_spawn = model['d_spawn']
        self.Q_spawn = model['Q_spawn']
        self.birth_GM = model['birth_GM']
        self.p_d = model['p_d']
        self.H = model['H']
        self.R = model['R']
        self.clutter_density_func = model['clutt_int_fun']
        self.T = model['T']
        self.U = model['U']
        self.Jmax = model['Jmax']

    def thinning_and_displacement(self, v: GaussianMixture, p, F: np.ndarray, Q: np.ndarray):
        """
        For the given Gaussian mixture v, perform thinning with probability P and displacement with N(x; F @ x_prev, Q)
        """
        w = []
        m = []
        P = []
        for weight in v.w:
            w.append(weight * p)
        for mean in v.m:
            m.append(F @ mean)
        for cov_matrix in v.P:
            P.append(Q + F @ cov_matrix @ F.T)
        return GaussianMixture(w, m, P)

    def spawn_mixture(self, v):
        """
        Spawning targets in prediction step
        """
        w = []
        m = []
        P = []
        for i, w_v in enumerate(v.w):
            for j, w_spawn in enumerate(self.w_spawn):
                w.append(w_v * w_spawn)
                m.append(self.F_spawn[j] @ v.m[i] + self.d_spawn[j])
                P.append(self.Q_spawn[j] + self.F_spawn[j] @ v.P[i] @ self.F_spawn[j].T)
        return GaussianMixture(w, m, P)

    def get_list_of_determinants(self, P_list):
        detP = []
        for P in P_list:
            detP.append(lin.det(P))
        return detP

    def get_list_of_inverses(self, P_list):
        invP = []
        for P in P_list:
            invP.append(lin.inv(P))
        return invP

    def prediction(self, v):
        """
        Prediction step of the GMPHD filter
        Inputs:
        - v: Gaussian mixture of the previous step
        """
        # v_pred = v_s + v_spawn +  v_new_born
        birth_copy = self.birth_GM.copy()
        # targets that survived v_s:
        v_s = self.thinning_and_displacement(v, self.p_s, self.F, self.Q)
        # spawning targets
        v_spawn = self.spawn_mixture(v)
        # final phd of prediction
        return GaussianMixture(v_s.w + v_spawn.w + birth_copy.w, v_s.m + v_spawn.m + birth_copy.m,
                               v_s.P + v_spawn.P + birth_copy.P)

    def correction(self, v: GaussianMixture, Z):
        """
        Correction step of the GMPHD filter
        Inputs:
        - v: Gaussian mixture obtained from the prediction step implemented in prediction function
        - Z: Measurement set, containing list of observations
        """
        v_residual = self.thinning_and_displacement(v, self.p_d, self.H, self.R)
        detP = self.get_list_of_determinants(v_residual.P)
        invP = self.get_list_of_inverses(v_residual.P)
        v_residual.assign_determinant_and_inverse(detP, invP)

        K = []
        P_kk = []
        for i in range(len(v_residual.w)):
            k = v.P[i] @ self.H.T @ invP[i]
            K.append(k)
            P_kk.append(v.P[i] - k @ self.H @ v.P[i])

        v_copy = v.copy()
        w = (np.array(v_copy.w) * (1 - self.p_d)).tolist()
        m = v_copy.m
        P = v_copy.P

        for z in Z:
            values = v_residual.mixture_component_values_list(z)
            normalization_factor = np.sum(values) + self.clutter_density_func(z)
            for i in range(len(v_residual.w)):
                w.append(values[i] / normalization_factor)
                m.append(v.m[i] + K[i] @ (z - v_residual.m[i]))
                P.append(P_kk[i].copy())

        return GaussianMixture(w, m, P)

    def pruning(self, v: GaussianMixture):
        I = (np.array(v.w) > self.T).nonzero()[0]
        w = [v.w[i] for i in I]
        m = [v.m[i] for i in I]
        P = [v.P[i] for i in I]
        v = GaussianMixture(w, m, P)
        I = (np.array(v.w) > self.T).nonzero()[0].tolist()
        invP = self.get_list_of_inverses(v.P)
        vw = np.array(v.w)
        vm = np.array(v.m)
        w = []
        m = []
        P = []
        while len(I) > 0:
            j = I[0]
            for i in I:
                if vw[i] > vw[j]:
                    j = i
            L = []
            for i in I:
                if (vm[i] - vm[j]) @ invP[i] @ (vm[i] - vm[j]) <= self.U:
                    L.append(i)
            w_new = np.sum(vw[L])
            m_new = np.sum((vw[L] * vm[L].T).T, axis=0) / w_new
            P_new = np.zeros((m_new.shape[0], m_new.shape[0]))
            for i in L:
                P_new += vw[i] * (v.P[i] + np.outer(m_new - vm[i], m_new - vm[i]))
            P_new /= w_new
            w.append(w_new)
            m.append(m_new)
            P.append(P_new)
            I = [i for i in I if i not in L]

        if len(w) > self.Jmax:
            L = np.array(w).argsort()[-self.Jmax:]
            w = [w[i] for i in L]
            m = [m[i] for i in L]
            P = [P[i] for i in L]

        return GaussianMixture(w, m, P)

    def state_estimation(self, v: GaussianMixture):
        X = []
        for i in range(len(v.w)):
            if v.w[i] >= 0.5:
                for j in range(int(np.round(v.w[i]))):
                    X.append(v.m[i])
        return X

    def filter_data(self, Z):
        """
        Input:
        -Z: list of lists of np.ndarray representing the observations for each time step
        Output:
        -X: list of lists of np.ndarray representing the estimations for each time step
        """
        X = []
        v = GaussianMixture([], [], [])
        for z in Z:
            v = self.prediction(v)
            v = self.correction(v, z)
            v = self.pruning(v)
            x = self.state_estimation(v)
            X.append(x)
        return X


def process_model_for_example_1():
    # This is the model for the example in "Bayesian Multiple Target Filtering Using Random Finite Sets" by Vo, Vo, Clark
    # The implementation almost analog to Matlab code provided by Vo in http://ba-tuong.vo-au.com/codes.html

    model = {}

    # Sampling time, time step duration
    T_s = 1.
    model['T_s'] = T_s

    # number of scans, number of iterations in our simulation
    model['num_scans'] = 100

    # Surveillance region
    x_min = -1000
    x_max = 1000
    y_min = -1000
    y_max = 1000
    model['surveillance_region'] = np.array([[x_min, x_max], [y_min, y_max]])

    # TRANSITION MODEL
    # Probability of survival
    model['p_s'] = 0.99

    # Transition matrix
    I_2 = np.eye(2)
    # F = [[I_2, T_s*I_2], [02, I_2]
    F = np.zeros((4, 4))
    F[0:2, 0:2] = I_2
    F[0:2, 2:] = I_2 * T_s
    F[2:, 2:] = I_2
    model['F'] = F

    # Process noise covariance matrix
    Q = np.zeros((4, 4))
    Q[0:2, 0:2] = (T_s ** 4) / 4 * I_2
    Q[0:2, 2:] = (T_s ** 3) / 2 * I_2
    Q[2:, 0:2] = (T_s ** 3) / 2 * I_2
    Q[2:, 2:] = (T_s ** 2) * I_2
    # standard deviation of the process noise
    sigma_w = 5.
    Q = Q * (sigma_w ** 2)
    model['Q'] = Q

    # Parameters for the spawning model: beta(x|ksi) = sum(w[i]*Normal(x,F_spawn[i]*ksi+d_spawn[i],Q_spawn[i]))
    model['F_spawn'] = []
    model['d_spawn'] = []
    model['Q_spawn'] = []
    model['w_spawn'] = []

    # Parameters of the new born targets Gaussian mixture
    w = [0.03] * 4
    m = [np.array([0., 0., 0., 0.]), np.array([400., -600., 0., 0.]), np.array([-800., -200., 0., 0.]),
         np.array([-200., 800., 0., 0.])]
    P_pom_ = np.diag([100., 100., 100., 100.])
    P = [P_pom_.copy(), P_pom_.copy(), P_pom_.copy(), P_pom_.copy()]
    model['birth_GM'] = GaussianMixture(w, m, P)

    # MEASUREMENT MODEL
    # probability of detection
    model['p_d'] = 0.98

    # measurement matrix z = Hx + v = N(z; Hx, R)
    model['H'] = np.zeros((2, 4))
    model['H'][:, 0:2] = np.eye(2)
    # measurement noise covariance matrix
    sigma_v = 10  # m
    model['R'] = I_2 * (sigma_v ** 2)

    # the reference to clutter intensity function
    model['lc'] = 50
    model['clutt_int_fun'] = lambda z: clutter_intensity_function(z, model['lc'], model['surveillance_region'])

    # pruning and merging parameters:
    model['T'] = 1e-5
    model['U'] = 4.
    model['Jmax'] = 100

    return model


def process_model_for_example_2():
    """
    This is the model of the process for the example in "Bayesian Multiple Target Filtering Using Random Finite Sets" by
    Vo, Vo, Clark. The model code is analog to Matlab code provided by
    Vo in http://ba-tuong.vo-au.com/codes.html

    :returns
    - model: dictionary containing the necessary parameters, read through code to understand it better
    """

    model = {}

    # Sampling time, time step duration
    T_s = 1.
    model['T_s'] = T_s

    # number of scans, number of iterations in our simulation
    model['num_scans'] = 100

    # Surveillance region
    x_min = -1000
    x_max = 1000
    y_min = -1000
    y_max = 1000
    model['surveillance_region'] = np.array([[x_min, x_max], [y_min, y_max]])

    # TRANSITION MODEL
    # Probability of survival
    model['p_s'] = 0.99

    # Transition matrix
    I_2 = np.eye(2)
    # F = [[I_2, T_s*I_2], [02, I_2]
    F = np.zeros((4, 4))
    F[0:2, 0:2] = I_2
    F[0:2, 2:] = I_2 * T_s
    F[2:, 2:] = I_2
    model['F'] = F

    # Process noise covariance matrix
    Q = np.zeros((4, 4))
    Q[0:2, 0:2] = (T_s ** 4) / 4 * I_2
    Q[0:2, 2:] = (T_s ** 3) / 2 * I_2
    Q[2:, 0:2] = (T_s ** 3) / 2 * I_2
    Q[2:, 2:] = (T_s ** 2) * I_2
    # standard deviation of the process noise
    sigma_w = 5.
    Q = Q * (sigma_w ** 2)
    model['Q'] = Q

    # Parameters for the spawning model: beta(x|ksi) = sum(w[i]*Normal(x,F_spawn[i]*ksi+d_spawn[i],Q_spawn[i]))
    model['w_spawn'] = [0.05]
    model['F_spawn'] = [np.eye(4)]
    model['d_spawn'] = [0.0]
    Q_spawn = np.eye(4) * 100
    Q_spawn[[2, 3], [2, 3]] = 400
    model['Q_spawn'] = [Q_spawn]

    # Parameters of the new born targets Gaussian mixture
    w = [0.1, 0.1]
    m = [np.array([250., 250., 0., 0.]), np.array([-250., -250., 0., 0.])]
    P = [np.diag([100., 100., 25., 25.]), np.diag([100., 100., 25., 25.])]
    model['birth_GM'] = GaussianMixture(w, m, P)

    # MEASUREMENT MODEL
    # probability of detection
    model['p_d'] = 0.98

    # measurement matrix z = Hx + v = N(z; Hx, R)
    model['H'] = np.zeros((2, 4))
    model['H'][:, 0:2] = np.eye(2)
    # measurement noise covariance matrix
    sigma_v = 10  # m
    model['R'] = I_2 * (sigma_v ** 2)

    # the reference to clutter intensity function
    model['lc'] = 50
    model['clutt_int_fun'] = lambda z: clutter_intensity_function(z, model['lc'], model['surveillance_region'])

    # pruning and merging parameters:
    model['T'] = 1e-5
    model['U'] = 4.
    model['Jmax'] = 100

    return model


def extract_positions_of_targets(X_collection):
    X_pos = []
    for X_set in X_collection:
        x = []
        for state in X_set:
            x.append(state[0:2])
        X_pos.append(x)
    return X_pos


def example1(num_of_scans=100):
    targets_birth_time = [1, 1, 1, 20, 20, 20, 40, 40, 60, 60, 80, 80]
    targets_birth_time = (np.array(targets_birth_time) - 1).tolist()
    targets_death_time = [70, num_of_scans, 70, num_of_scans, num_of_scans, num_of_scans,
                          num_of_scans, num_of_scans, num_of_scans, num_of_scans, num_of_scans,
                          num_of_scans]
    targets_start = [np.array([0., 0., 0., -10.]),
                     np.array([400., -600., -10., 5.]),
                     np.array([-800., -200., 20., -5.]),

                     np.array([400., -600., -7., -4.]),
                     np.array([400., -600., -2.5, 10.]),
                     np.array([0., 0., 7.5, -5.]),

                     np.array([-800., -200., 12., 7.]),
                     np.array([-200., 800., 15., -10.]),

                     np.array([-800., -200., 3., 15.]),
                     np.array([-200., 800., -3., -15.]),

                     np.array([0., 0., -20., -15.]),
                     np.array([-200., 800., 15., -5.])]
    return targets_birth_time, targets_death_time, targets_start


def example2(num_of_scans=100):
    targets_birth_time = [1, 1]
    targets_birth_time = (np.array(targets_birth_time) - 1).tolist()
    targets_death_time = [num_of_scans, num_of_scans]
    targets_start = [np.array([250., 250., 2.5, -11.5]),
                     np.array([-250., -250., 11.5, -2.5])]
    # for spawning targets, there is birth time, death time, initial velocity and target from which it spawns
    targets_spw_time_brttgt_vel = [(66, num_of_scans, np.array([-20., 4.]), 0)]

    return targets_birth_time, targets_death_time, targets_start, targets_spw_time_brttgt_vel


def generate_trajectories(model, targets_birth_time, targets_death_time, targets_start, targets_spw_time_brttgt_vel=[],
                          noise=False):
    num_of_scans = model['num_scans']
    trajectories = []
    for i in range(num_of_scans):
        trajectories.append([])
    targets_tracks = {}
    for i, start in enumerate(targets_start):
        target_state = start
        targets_tracks[i] = []
        for k in range(targets_birth_time[i], min(targets_death_time[i], num_of_scans)):
            target_state = model['F'] @ target_state
            if noise:
                target_state += np.random.multivariate_normal(np.zeros(target_state.size), model['Q'])
            if target_state[0] < model['surveillance_region'][0][0] or target_state[0] > \
                    model['surveillance_region'][0][1] or target_state[1] < model['surveillance_region'][1][0] or \
                    target_state[1] > model['surveillance_region'][1][1]:
                targets_death_time[i] = k - 1
                break
            trajectories[k].append(target_state)
            targets_tracks[i].append(target_state)
    # next part is only for spawning targets. In examples, this part is often omitted.
    for i, item in enumerate(targets_spw_time_brttgt_vel):
        (target_birth_time, target_death_time, velocity, parent) = item
        target_state = np.zeros(4)
        if target_birth_time - targets_birth_time[parent] < 0 or target_death_time - targets_death_time[parent] > 0:
            continue
        target_state[0:2] = targets_tracks[parent][target_birth_time - targets_birth_time[parent]][0:2]
        target_state[2:] = velocity
        targets_birth_time.append(target_birth_time)
        targets_death_time.append(target_death_time)
        targets_start.append(target_state)
        # trajectories.append([])
        targets_tracks[len(targets_birth_time) - 1] = []
        for k in range(target_birth_time, min(target_death_time, num_of_scans)):
            target_state = model['F'] @ target_state
            if noise:
                target_state += np.random.multivariate_normal(np.zeros(target_state.size), model['Q'])
            if target_state[0] < model['surveillance_region'][0][0] or target_state[0] > \
                    model['surveillance_region'][0][1] or target_state[1] < model['surveillance_region'][1][0] or \
                    target_state[1] > model['surveillance_region'][1][1]:
                targets_death_time[-1] = k - 1
                break
            trajectories[k].append(target_state)
            targets_tracks[len(targets_birth_time) - 1].append(target_state)
    return trajectories, targets_tracks


def generate_measurements(model, trajectories):
    data = []
    surveillanceRegion = model['surveillance_region']
    for X in trajectories:
        m = []
        for state in X:
            if np.random.rand() <= model['p_d']:
                meas = model['H'] @ state + np.random.multivariate_normal(np.zeros(model['H'].shape[0]), model['R'])
                m.append(meas)
        for i in range(np.random.poisson(model['lc'])):
            x = (surveillanceRegion[0][1] - surveillanceRegion[0][0]) * np.random.rand() + surveillanceRegion[0][0]
            y = (surveillanceRegion[1][1] - surveillanceRegion[1][0]) * np.random.rand() + surveillanceRegion[1][0]
            m.append(np.array([x, y]))
        data.append(m)
    return data


def true_trajectory_tracks_plots(targets_birth_time, targets_tracks, delta):
    for_plot = {}
    for i, birth in enumerate(targets_birth_time):
        brojac = birth
        x = []
        y = []
        time = []
        for state in targets_tracks[i]:
            x.append(state[0])
            y.append(state[1])
            time.append(brojac)
            brojac += delta
        for_plot[i] = (time, x, y)
    return for_plot


def extract_axis_for_plot(X_collection, delta):
    time = []
    x = []
    y = []
    k = 0
    for X in X_collection:
        for state in X:
            x.append(state[0])
            y.append(state[1])
            time.append(k)
        k += delta
    return time, x, y


if __name__ == '__main__':

    # For example 1, uncomment the following code.
    # =================================================Example 1========================================================
    # model = process_model_for_example_1()
    # targets_birth_time, targets_death_time, targets_start = example1(model['num_scans'])
    # trajectories, targets_tracks = generate_trajectories(model, targets_birth_time, targets_death_time, targets_start,
    #                                                      noise=False)
    # ==================================================================================================================

    # For example 2, uncomment the following code.
    # =================================================Example 2========================================================
    model = process_model_for_example_2()
    targets_birth_time, targets_death_time, targets_start, targets_spw_time_brttgt_vel = example2(model['num_scans'])
    trajectories, targets_tracks = generate_trajectories(model, targets_birth_time, targets_death_time, targets_start,
                                                         targets_spw_time_brttgt_vel, noise=False)
    # ==================================================================================================================

    # Collections of observations for each time step
    data = generate_measurements(model, trajectories)

    # Call of the gmphd filter on the created observations collections
    gmphd = GmphdFilter(model)
    a = time.time()
    X_collection = gmphd.filter_data(data)
    print('Filtration time: ' + str(time.time() - a) + ' sec')

    # Plotting the results of filtration saved in X_collection file
    tracks_plot = true_trajectory_tracks_plots(targets_birth_time, targets_tracks, model['T_s'])
    plt.figure()
    for key in tracks_plot:
        t, x, y = tracks_plot[key]
        plt.plot(x[0], y[0], 'o', c='k', mfc='none')
        plt.plot(x[-1], y[-1], 's', c='k', mfc='none')
        plt.plot(x, y)
    plt.axis(model['surveillance_region'].flatten())
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.title(r"Targets movement in surveilance region. Circle represents the starting point and"
    #           r" square represents the end point.",loc='center', wrap=True)

    # plot measurements, true trajectories and estimations
    meas_time, meas_x, meas_y = extract_axis_for_plot(data, model['T_s'])
    estim_time, estim_x, estim_y = extract_axis_for_plot(X_collection, model['T_s'])
    plt.figure()
    plt.plot(meas_time, meas_x, 'x', c='C0')
    for key in tracks_plot:
        t, x, y = tracks_plot[key]
        plt.plot(t, x, 'r')
    plt.plot(estim_time, estim_x, 'o', c='k', markersize=3)
    plt.xlabel('time[$sec$]')
    plt.ylabel('x')

    # plot measurements, true trajectories and estimations
    plt.figure()
    plt.plot(meas_time, meas_y, 'x', c='C0')
    for key in tracks_plot:
        t, x, y = tracks_plot[key]
        plt.plot(t, y, 'r')
    plt.plot(estim_time, estim_y, 'o', c='k', markersize=3)
    plt.xlabel('time[$sec$]')
    plt.ylabel('y')

    num_targets_truth = []
    num_targets_estimated = []

    for x_set in trajectories:
        num_targets_truth.append(len(x_set))
    for x_set in X_collection:
        num_targets_estimated.append(len(x_set))

    plt.figure()
    (markerline, stemlines, baseline) = plt.stem(num_targets_estimated, label='estimated number of targets')
    plt.setp(baseline, color='k')  # visible=False)
    plt.setp(stemlines, visible=False)  # visible=False)
    plt.setp(markerline, markersize=3.0)
    plt.step(num_targets_truth, 'r', label='actual number of targets')
    plt.xlabel('time[$sec$]')
    plt.legend()
    plt.title('Estimated cardinality VS actual cardinality')
    # plt.show()
