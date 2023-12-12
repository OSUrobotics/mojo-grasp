import data_gui_backend
import matplotlib.pyplot as plt
thing = data_gui_backend.PlotBackend('./demos/rl_demo/data/JA_fullstate_A_rand')

thing.clear_plots = False
thing.moving_avg =500
thing.draw_net_reward('./demos/rl_demo/data/JA_fullstate_A_rand/Test/')
thing.draw_net_reward('./demos/rl_demo/data/FTP_fullstate_A_rand/Test/')
thing.draw_net_reward('./demos/rl_demo/data/JA_halfstate_A_rand/Test/')
thing.draw_net_reward('./demos/rl_demo/data/FTP_halfstate_A_rand/Test/')

plt.show()