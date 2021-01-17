from gmphd import *
from examples import *
import pickle
import ospa

def MC_run():
    # parameters for OSPA metric
    c = 100.
    p = 1

    model = process_model_for_example_1()
    targets_birth_time, targets_death_time, targets_start = example1(100)
    trajectories, targets_tracks = generate_trajectories(model, targets_birth_time, targets_death_time, targets_start,
                                                         noise=False)
    truth_collection = extract_positions_of_targets(trajectories)

    osp_all_total = []
    osp_loc_total = []
    osp_card_total = []
    tnum_total = []
    for i in range(1000):
        a = time.time()
        data = generate_measurements(model, trajectories)
        gmphd = GmphdFilter(model)
        X_collection = gmphd.filter_data(data)
        X_pos = extract_positions_of_targets(X_collection)
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

    with open('MC2ospatnum1000.pkl', 'wb') as output:
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
    # To run Monte Carlo simulations, uncomment the following function call
    # ====================Monte Carlo Simulation=====================================
    # MC_run()
    # ===============================================================================

    # If you want to plot results of MC_run() function that are saved in file,
    # uncomment the following section
    # ====================Monte Carlo Plot results===================================
    with open('MC2ospatnum1000.pkl', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        (osp_all_total, osp_loc_total, osp_card_total, tnum_total) = pickle.load(f)
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
        plt.xlabel('Time step')
        plt.ylabel('OSPA Distance')
        plt.grid(True)

        plt.figure()
        plt.plot(ospLocMean)
        plt.xlabel('Time step')
        plt.ylabel('OSPA Localization')
        plt.grid(True)

        plt.figure()
        plt.plot(ospCarMean)
        plt.xlabel('Time step')
        plt.ylabel('OSPA Cardinality')
        plt.grid(True)

        plt.figure()
        plt.plot(tnumMean)
        plt.plot(tnumMean - tnumStd)
        plt.plot(tnumMean + tnumStd)
        plt.xlabel('Time step')
        plt.ylabel('Avg. numb. of targets')
        plt.grid(True)
        plt.title('Average number of targets with standard deviation')
        plt.show()
    # ===============================================================================
