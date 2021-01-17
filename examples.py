from gmphd import *

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
    model = process_model_for_example_1()
    targets_birth_time, targets_death_time, targets_start = example1(model['num_scans'])
    trajectories, targets_tracks = generate_trajectories(model, targets_birth_time, targets_death_time, targets_start,
                                                         noise=False)
    # ==================================================================================================================

    # For example 2, uncomment the following code.
    # =================================================Example 2========================================================
    # model = process_model_for_example_2()
    # targets_birth_time, targets_death_time, targets_start, targets_spw_time_brttgt_vel = example2(model['num_scans'])
    # trajectories, targets_tracks = generate_trajectories(model, targets_birth_time, targets_death_time, targets_start,
    #                                                      targets_spw_time_brttgt_vel, noise=False)
    # ==================================================================================================================

    # Collections of observations for each time step
    data = generate_measurements(model, trajectories)

    # Call of the gmphd filter for the created observations collections
    gmphd = GmphdFilter(model)
    a = time.time()
    X_collection = gmphd.filter_data(data)
    print('Filtration time: ' + str(time.time() - a) + ' sec')

    # Plot the results of filtration saved in X_collection file
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
    plt.title(r"Targets movement in surveilance region. Circle represents the starting point and"
              r" square represents the end point.", loc='center', wrap=True)

    # Plot measurements, true trajectories and estimations
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
    plt.title('X axis in time. Blue x are measurements(50 in each time step), '
              'black dots are estimations and the red lines are actual trajectories of targets', loc='center', wrap=True)

    plt.figure()
    plt.plot(meas_time, meas_y, 'x', c='C0')
    for key in tracks_plot:
        t, x, y = tracks_plot[key]
        plt.plot(t, y, 'r')
    plt.plot(estim_time, estim_y, 'o', c='k', markersize=3)
    plt.xlabel('time[$sec$]')
    plt.ylabel('y')
    plt.title('Y axis in time. Blue x are measurements(50 in each time step), '
              'black dots are estimations and the red lines are actual trajectories of targets', loc='center', wrap=True)

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
    plt.title('Estimated cardinality VS actual cardinality', loc='center', wrap=True)
    plt.show()
