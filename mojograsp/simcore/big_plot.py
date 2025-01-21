import data_gui_backend
import matplotlib.pyplot as plt
thing = data_gui_backend.PlotBackend('./demos/rl_demo/data/JA_fullstate_A_rand')

thing.clear_plots = False
thing.moving_avg =600
JA_full = thing.draw_average_efficiency('./demos/rl_demo/data/JA_fullstate_A_rand/Test/')
FTP_full = thing.draw_average_efficiency('./demos/rl_demo/data/FTP_fullstate_A_rand/Test/')
JA_half = thing.draw_average_efficiency('./demos/rl_demo/data/JA_halfstate_A_rand/Test/')
FTP_half = thing.draw_average_efficiency('./demos/rl_demo/data/FTP_halfstate_A_rand/Test/')

# JA_full_b = thing.draw_ending_goal_dist('./demos/rl_demo/data/JA_fullstate_A_rand/eval_b_moving/')
# FTP_full_b = thing.draw_ending_goal_dist('./demos/rl_demo/data/FTP_fullstate_A_rand/eval_b_moving/')
# JA_half_b = thing.draw_ending_goal_dist('./demos/rl_demo/data/JA_halfstate_A_rand/eval_b_moving/')
# FTP_half_b = thing.draw_ending_goal_dist('./demos/rl_demo/data/FTP_halfstate_A_rand/eval_b_moving/')
plt.legend(['JA Full','FTP Full', 'JA Partial','FTP Partial'])


# print('fullstate FTP', FTP_full, FTP_full_b)
# print('fullstate JA', JA_full, JA_full_b)
# print('halfstate FTP', FTP_half, FTP_half_b)
# print('halfstate JA', JA_half, JA_half_b)

plt.show()