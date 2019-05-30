import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lin
import time
import ospa
import pickle


def pdf_multivariate_gauss(x, mu, cov):
    '''
    Caculate the multivariate normal density (pdf)

    Keyword arguments:
        x = numpy array of a "d x 1" sample vector
        mu = numpy array of a "d x 1" mean vector
        cov = "numpy array of a d x d" covariance matrix
    '''

    part1 = 1 / (((2 * np.pi) ** (mu.size / 2)) * (lin.det(cov) ** 0.5))
    part2 = (-1 / 2) * ((x - mu) @ lin.inv(cov) @ (x - mu))
    return float(part1 * np.exp(part2))


def pdf_multivariate_gauss2(x, mu, detC, invC):
    """

    :param x: numpy array of a "d x 1" sample vector
    :param mu: numpy array of a "d x 1" mean vector
    :param detC: "numpy array of a d x d" determinant of covariance matrix
    :param invC: "numpy array of a d x d" inversion of covariance matrix
    :return:
    """
    part1 = 1 / (((2 * np.pi) ** (mu.size / 2)) * (detC ** 0.5))
    part2 = (-1 / 2) * ((x - mu) @ invC @ (x - mu))
    return float(part1 * np.exp(part2))


class GaussianMixture:
    def __init__(self, w, m, P):
        """
        The Gaussian mixture
        :param w: list of weights (list of scalar values)
        :param m: list of means (list of elements type ndarray)
        :param P: list of covariance matrices(list of elements type ndarray)
        """
        self.w = w
        self.m = m
        self.P = P

    def compute_density(self, x):
        my_sum = 0
        for i in range(len(self.w)):
            my_sum += self.w[i] * pdf_multivariate_gauss(x, self.m[i], self.P[i])
        return my_sum

    def copy(self):
        """
        :return:Deep copy of object
        """
        w = self.w.copy()
        m = []
        P = []
        for mean in self.m:
            m.append(mean.copy())
        for cov in self.P:
            P.append(cov.copy())
        return GaussianMixture(w, m, P)


class GMPHD:
    def __init__(self, model):
        """
        The Gaussian Mixture Probability Hypothesis Density filter implementation. It's based on
        "The Gaussian mixture probability hypothesis density filter" by Vo and Ma.
        Note that x will be 1D ndarray.
            x[k] = Fx[k-1] + w[k-1]
            y[k] = Hx[k] + v[k]
        :param model: dictionary which contains the following elements(keys are strings):
               F: state transition matrix
               H:
               Q: process noise covariance matrix(of variable w[k]). If it's scalar, you should pass scalar
               R: measurement noise covariance matrix(of variable v[k]). If it's scalar, you should pass scalar
             p_d: probability of target detection
             p_s: probability of target survival

         Spawning model, see paper pg. 5. it's a gaussian mixture conditioned on state
         F_spawn:  d_spawn: Q_spawn: w_spawn: lists with the same size, see pg. 5

    clutt_int_fun: reference to clutter intensity function, gets only one argument, which is the current measure

               T: U: Jmax: Pruning parameters, see pg. 7.

        birth_GM: The Gaussian Mixture of the birth intensity
        """
        self.F = model['F']
        self.H = model['H']
        self.Q = model['Q']
        self.R = model['R']

        # Parameters for the spawning model: beta(x|ksi) = sum(w[i]*Normal(x,F_spawn[i]*ksi+d_spawn[i],Q_spawn[i]))
        self.F_spawn = model['F_spawn']
        self.d_spawn = model['d_spawn']
        self.Q_spawn = model['Q_spawn']
        self.w_spawn = model['w_spawn']

        # birth Gaussian mixture
        self.birth_GM = model['birth_GM']

        # probability of survival and detection
        self.p_d = model['p_d']
        self.p_s = model['p_s']

        # the reference to clutter intensity function
        self.clutter_intensity = model['clutt_int_fun']

        # pruning and merging parameters:
        self.T = model['T']
        self.U = model['U']
        self.Jmax = model['Jmax']

        # intensity function of prediction and correction. Estimated state set X for current iteration
        self.v_pred = None
        self.v_corr = GaussianMixture([], [], [])
        self.X = None

    def prediction(self):
        # prediction for birth targets
        gm = self.birth_GM.copy()
        w = gm.w
        m = gm.m
        P = gm.P

        # prediction for spawning targets
        for j, wspwn in enumerate(self.w_spawn):
            for l, wcorr in enumerate(self.v_corr.w):
                w.append(wspwn * wcorr)
                m.append(self.F_spawn[j] @ self.v_corr.m[l] + self.d_spawn[j])
                P.append(self.Q_spawn[j] + self.F_spawn[j] @ self.v_corr.P[l] @ self.F_spawn[j].T)

        # prediction for existing targets
        w.extend(np.array(self.v_corr.w) * self.p_s)
        for mean in self.v_corr.m:
            m.append(self.F @ mean)
        for Pc in self.v_corr.P:
            P.append(self.Q + self.F @ Pc @ self.F.T)

        self.v_pred = GaussianMixture(w, m, P)

    def correction(self, z_set):
        eta = []
        S = []
        detS = []
        invS = []
        K = []
        Pk = []
        for mean in self.v_pred.m:
            eta.append(self.H @ mean)
        for P1 in self.v_pred.P:
            s = self.R + self.H @ P1 @ self.H.T
            S.append(s)
            detS.append(lin.det(s))
            invS.append(lin.inv(s))
            K.append(P1 @ self.H.T @ invS[-1])
            Pk.append(P1 - K[-1] @ self.H @ P1)
        pm = self.v_pred.copy()
        w = (np.array(pm.w) * (1 - self.p_d)).tolist()
        m = pm.m
        P = pm.P
        for z in z_set:
            w1 = []
            for j, wpred in enumerate(self.v_pred.w):
                w1.append(self.p_d * wpred * pdf_multivariate_gauss2(z, eta[j], detS[j], invS[j]))
                m.append(self.v_pred.m[j] + K[j] @ (z - eta[j]))
                P.append(Pk[j].copy())
            w1 = np.array(w1)
            c1 = self.clutter_intensity(z) + w1.sum()
            w1 = w1 / c1
            w.extend(w1)
        self.v_corr = GaussianMixture(w, m, P)

    def prune(self):
        I = (np.array(self.v_corr.w) > self.T).nonzero()[0]
        w = [self.v_corr.w[i] for i in I]
        m = [self.v_corr.m[i] for i in I]
        P = [self.v_corr.P[i] for i in I]
        self.v_corr = GaussianMixture(w, m, P)

    def merge(self):
        w = []
        m = []
        P = []
        invP = []
        for P1 in self.v_corr.P:
            invP.append(lin.inv(P1))
        I = np.array(self.v_corr.w).nonzero()[0].tolist()
        while len(I) > 0:
            j = I[0]
            for i in I:
                if self.v_corr.w[i] > self.v_corr.w[j]:
                    j = i
            L = []
            for i in I:
                if (self.v_corr.m[i] - self.v_corr.m[j]).T @ invP[i] @ (self.v_corr.m[i] - self.v_corr.m[j]) <= self.U:
                    L.append(i)
            # w_new = np.array(self.v_corr.w)[L].sum()
            w_new = 0
            for i in L:
                w_new += self.v_corr.w[i]
            m_new = np.zeros(self.v_corr.m[0].shape)
            P_new = np.zeros(self.v_corr.P[0].shape)
            for i in L:
                m_new += self.v_corr.w[i] * self.v_corr.m[i]
            m_new = m_new / w_new
            for i in L:
                P_new += self.v_corr.w[i] * (
                        self.v_corr.P[i] + np.outer(m_new - self.v_corr.m[i], m_new - self.v_corr.m[i]))
            P_new = P_new / w_new
            w.append(w_new)
            m.append(m_new)
            P.append(P_new)
            I = [i for i in I if i not in L]
        if len(w) > self.Jmax:
            L = np.array(w).argsort()[-self.Jmax:]
            w = [w[i] for i in L]
            m = [m[i] for i in L]
            P = [P[i] for i in L]
        self.v_corr = GaussianMixture(w, m, P)

    def state_extraction(self):
        x = []
        for i, wght in enumerate(self.v_corr.w):
            if wght > 0.5:
                for j in range(int(round(wght))):
                    x.append(self.v_corr.m[i])
        self.X = x
        return x

    def filter(self, z_set):
        self.prediction()
        self.correction(z_set)
        self.prune()
        self.merge()
        x = self.state_extraction()
        return x

    def run_filter(self, data):
        X_collection = []
        for z_set in data:
            X_collection.append(self.filter(z_set))
        return X_collection


def plot_results():
    pass


def extract_position_collection(X_collection):
    X_pos = []
    for X_set in X_collection:
        x = []
        for state in X_set:
            x.append(state[0:2])
        X_pos.append(x)
    return X_pos


def clutter_intensity_function(pos, lc, surveillance_region):
    '''
    Clutter intensity function, with uniform distribution through the surveillance region, see pg. 8
    :param pos:
    :param lc:
    :param surveillance_region:
    '''
    if surveillance_region[0] <= pos[0] <= surveillance_region[1] and surveillance_region[2] <= pos[1] <= \
            surveillance_region[3]:
        return lc / ((surveillance_region[1] - surveillance_region[0]) * (
                surveillance_region[3] - surveillance_region[2]))
    else:
        return 0


def generate_model():
    # This is the model for the example in "Bayesian Multiple Target Filtering Using Random Finite Sets" by Vo, Vo, Clark
    # The implementation almost analog to Matlab code provided by Vo in http://ba-tuong.vo-au.com/codes.html

    # surveillance region
    xmin = -1000
    xmax = 1000
    ymin = -1000
    ymax = 1000

    # model - model of system
    model = {}
    model['surveillance_region'] = np.array([xmin, xmax, ymin, ymax])
    # Sampling time
    delta = 1.
    model['delta'] = delta
    model['num_scans'] = 100
    # F = [[I2, delta*I2], [02, I2]
    F = np.zeros((4, 4))
    F[0:2, 0:2] = np.eye(2, 2)
    F[0:2, 2:] = np.eye(2, 2) * delta
    F[2:, 2:] = np.eye(2, 2)
    model['F'] = F

    sv = 5.
    Q = np.zeros((4, 4))
    Q[0:2, 0:2] = (delta ** 4) / 4 * np.eye(2, 2)
    Q[0:2, 2:] = (delta ** 3) / 2 * np.eye(2, 2)
    Q[2:, 0:2] = (delta ** 3) / 2 * np.eye(2, 2)
    Q[2:, 2:] = (delta ** 2) * np.eye(2, 2)
    Q = Q * (sv ** 2)
    model['Q'] = Q

    # Parameters for the spawning model: beta(x|ksi) = sum(w[i]*Normal(x,F_spawn[i]*ksi+d_spawn[i],Q_spawn[i]))
    model['F_spawn'] = []
    model['d_spawn'] = []
    model['Q_spawn'] = []
    model['w_spawn'] = []

    # probability of survival and detection
    model['p_d'] = 0.98
    model['p_s'] = 0.99

    w = [0.03] * 4
    m = [np.array([0., 0., 0., 0.]), np.array([400., -600., 0., 0.]), np.array([-800., -200., 0., 0.]),
         np.array([-200., 800., 0., 0.])]
    Ppom = np.diag([100., 100., 100., 100.])
    P = [Ppom.copy(), Ppom.copy(), Ppom.copy(), Ppom.copy()]
    model['birth_GM'] = GaussianMixture(w, m, P)

    model['H'] = np.zeros((2, 4))
    model['H'][:, 0:2] = np.eye(2)
    se = 10  # m
    model['R'] = np.eye(2) * (se ** 2)

    # the reference to clutter intensity function
    model['lc'] = 50
    model['clutt_int_fun'] = lambda z: clutter_intensity_function(z, model['lc'], model['surveillance_region'])

    # pruning and merging parameters:
    model['T'] = 1e-5
    model['U'] = 4.
    model['Jmax'] = 100

    return model


def generate_model2():
    # This is the model for the example in "The Gaussian mixture probability hypothesis density filter" by Vo and Ma.

    # surveillance region
    xmin = -1000
    xmax = 1000
    ymin = -1000
    ymax = 1000

    # model - model of system
    model = {}
    model['surveillance_region'] = np.array([xmin, xmax, ymin, ymax])
    # Sampling time
    delta = 1.
    model['delta'] = delta
    model['num_scans'] = 100
    # F = [[I2, delta*I2], [02, I2]
    F = np.zeros((4, 4))
    F[0:2, 0:2] = np.eye(2, 2)
    F[0:2, 2:] = np.eye(2, 2) * delta
    F[2:, 2:] = np.eye(2, 2)
    model['F'] = F

    sv = 5.
    Q = np.zeros((4, 4))
    Q[0:2, 0:2] = (delta ** 4) / 4 * np.eye(2, 2)
    Q[0:2, 2:] = (delta ** 3) / 2 * np.eye(2, 2)
    Q[2:, 0:2] = (delta ** 3) / 2 * np.eye(2, 2)
    Q[2:, 2:] = (delta ** 2) * np.eye(2, 2)
    Q = Q * (sv ** 2)
    model['Q'] = Q

    # Parameters for the spawning model: beta(x|ksi) = sum(w[i]*Normal(x,F_spawn[i]*ksi+d_spawn[i],Q_spawn[i]))
    model['F_spawn'] = [np.eye(4)]
    model['d_spawn'] = [np.zeros(4)]
    model['Q_spawn'] = [np.diag([100., 100, 400, 400])]
    model['w_spawn'] = [0.05]

    # probability of survival and detection
    model['p_d'] = 0.98
    model['p_s'] = 0.99

    # these are parameters for the example from GMPHD paper
    w = [0.1, 0.1]
    m = [np.array([250., 250., 0., 0.]), np.array([-250., -250., 0., 0.])]
    P = [np.diag([100., 100, 25, 25]), np.diag([100., 100, 25, 25])]

    model['birth_GM'] = GaussianMixture(w, m, P)

    model['H'] = np.zeros((2, 4))
    model['H'][:, 0:2] = np.eye(2)
    se = 10  # m
    model['R'] = np.eye(2) * (se ** 2)

    # the reference to clutter intensity function
    model['lc'] = 50
    model['clutt_int_fun'] = lambda z: clutter_intensity_function(z, model['lc'], model['surveillance_region'])

    # pruning and merging parameters:
    model['T'] = 1e-5
    model['U'] = 4.
    model['Jmax'] = 100

    return model


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
            if target_state[0] < model['surveillance_region'][0] or target_state[0] > model['surveillance_region'][1] or \
                    target_state[1] < model['surveillance_region'][2] or target_state[1] > model['surveillance_region'][
                3]:
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
            if target_state[0] < model['surveillance_region'][0] or target_state[0] > model['surveillance_region'][1] or \
                    target_state[1] < model['surveillance_region'][2] or target_state[1] > model['surveillance_region'][
                3]:
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
            x = (surveillanceRegion[1] - surveillanceRegion[0]) * np.random.rand() + surveillanceRegion[0]
            y = (surveillanceRegion[3] - surveillanceRegion[2]) * np.random.rand() + surveillanceRegion[2]
            m.append(np.array([x, y]))
        data.append(m)
    return data


def true_tracks_plots(targets_birth_time, targets_death_time, targets_tracks, delta):
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


def sets_collection_for_plot(X_collection, delta):
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


def MC_run():
    # parameters for OSPA metric
    c = 100.
    p = 1

    model = generate_model()
    targets_birth_time, targets_death_time, targets_start = example1(100)
    trajectories, targets_tracks = generate_trajectories(model, targets_birth_time, targets_death_time, targets_start,
                                                         noise=False)
    truth_collection = extract_position_collection(trajectories)

    osp_all_total = []
    osp_loc_total = []
    osp_card_total = []
    tnum_total = []
    for i in range(500):
        a = time.time()
        data = generate_measurements(model, trajectories)
        gmphd = GMPHD(model)
        X_collection = gmphd.run_filter(data)
        X_pos = extract_position_collection(X_collection)
        ospa_loc = []
        ospa_card = []
        ospa_all = []
        tnum = []
        for j, x_set in enumerate(X_pos):
            oall, oloc, ocard = ospa.ospa_all(truth_collection[j], x_set, c, p)
            ospa_all.append(oall)
            ospa_loc.append(oloc)
            ospa_card.append(ocard)
            tnum.append(len(x_set))
        osp_all_total.append(ospa_all)
        osp_loc_total.append(ospa_loc)
        osp_card_total.append(ospa_card)
        tnum_total.append(tnum)
        print('Iteration ' + str(i) + ', total time: ' + str(time.time() - a))

    with open('MC2ospatnum500.pkl', 'wb') as output:
        pickle.dump((osp_all_total, osp_loc_total, osp_card_total, tnum_total), output)

    ospall = np.array(osp_all_total)
    ospAllMean = ospall.mean(0)
    osploc = np.array(osp_loc_total)
    ospLocMean = osploc.mean(0)
    ospcar = np.array(osp_card_total)
    ospCarMean = ospcar.mean(0)
    tnum = np.array(tnum_total)
    tnumMean = tnum.mean(0)
    tnumStd = tnum.std(0)

    plt.figure()
    plt.plot(ospAllMean)

    plt.figure()
    plt.plot(ospLocMean)

    plt.figure()
    plt.plot(ospCarMean)

    plt.figure()
    plt.plot(tnumMean)
    plt.plot(tnumMean - tnumStd)
    plt.plot(tnumMean + tnumStd)


if __name__ == '__main__':
    model = generate_model()
    targets_birth_time, targets_death_time, targets_start = example1(100)
    trajectories, targets_tracks = generate_trajectories(model, targets_birth_time, targets_death_time, targets_start,
                                                         noise=False)
    # model = generate_model2()
    # targets_birth_time, targets_death_time, targets_start, targets_spw_time_brttgt_vel = example2(100)
    # trajectories, targets_tracks = generate_trajectories(model, targets_birth_time, targets_death_time, targets_start,
    #                                                      targets_spw_time_brttgt_vel, noise=False)
    data = generate_measurements(model, trajectories)
    gmphd = GMPHD(model)
    a = time.time()
    X_collection = gmphd.run_filter(data)
    print('Filtration time: ' + str(time.time() - a) + ' sec')

    # plot trajectories
    tracks_plot = true_tracks_plots(targets_birth_time, targets_death_time, targets_tracks, model['delta'])
    plt.figure()
    for key in tracks_plot:
        t, x, y = tracks_plot[key]
        plt.plot(x[0], y[0], 'o', c='k', mfc='none')
        plt.plot(x[-1], y[-1], 's', c='k', mfc='none')
        plt.plot(x, y)
    plt.axis(model['surveillance_region'])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x')
    plt.ylabel('y')

    # plot measurements, true trajectories and estimations
    meas_time, meas_x, meas_y = sets_collection_for_plot(data, model['delta'])
    estim_time, estim_x, estim_y = sets_collection_for_plot(X_collection, model['delta'])
    plt.figure()
    plt.plot(meas_time, meas_x, 'x', c='C0')
    for key in tracks_plot:
        t, x, y = tracks_plot[key]
        plt.plot(t, x, 'r')
    plt.plot(estim_time, estim_x, 'o', c='k', markersize=3)

    # plot measurements, true trajectories and estimations
    plt.figure()
    plt.plot(meas_time, meas_y, 'x', c='C0')
    for key in tracks_plot:
        t, x, y = tracks_plot[key]
        plt.plot(t, y, 'r')
    plt.plot(estim_time, estim_y, 'o', c='k', markersize=3)

    # MC_run()

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
    plt.xlabel('time($sec$)')
    plt.legend()
    plt.title('Estimated cardinality VS actual cardinality')