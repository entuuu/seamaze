import sys, matplotlib
matplotlib.use('Agg')
import runpy, logging
import numpy as np
g = runpy.run_path(sys.argv[1], run_name='__main__')
r = g['result']
x = r['optimal_point']
print('RESULT', sys.argv[1].split('\\')[-1], '| f =', r['optimal_value'], '| iter =', r['iterations'], '| time =', round(r['wall_time'], 2), 's | info =', r['solver_info'])
print('   x mean = %.3f, min = %.3f, max = %.3f (global optimum: 420.9687)' % (x.mean(), x.min(), x.max()))
