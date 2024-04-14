import matplotlib.pyplot as plt
import json
# plt.rcParams['font.sans-serif'] = [u'SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# Precision Recall Curve data
pr_data = {"audionet": "$path$/log/pr_data.txt"}

N = 150
# N = -1
for key in pr_data:
    path = pr_data[key]
    pr_data[key] = json.load(open(path))


# markers = "DdsPvo*xH1234h"
markers = ".........................."
method2marker = {}
i = 0
for method in pr_data:
    method2marker[method] = markers[i]
    i += 1

plt.figure(figsize=(15, 5))
plt.subplot(131)

for method in pr_data:
    P, R,draw_range = pr_data[method]["P"],pr_data[method]["R"],pr_data[method]["index"]
    print(len(P))
    print(len(R))
    plt.plot(R, P, linestyle="-", marker=method2marker[method], label=method)
plt.grid(True)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend()
plt.subplot(132)
for method in pr_data:
    P, R,draw_range = pr_data[method]["P"][:N],pr_data[method]["R"][:N],pr_data[method]["index"][:N]
    plt.plot(draw_range, R, linestyle="-", marker=method2marker[method], label=method)
plt.xlim(0, max(draw_range))
plt.grid(True)
plt.xlabel('The number of retrieved audio')
plt.ylabel('recall')
plt.legend()

plt.subplot(133)
for method in pr_data:
    P, R,draw_range = pr_data[method]["P"][:N],pr_data[method]["R"][:N],pr_data[method]["index"][:N]
    plt.plot(draw_range, P, linestyle="-", marker=method2marker[method], label=method)
plt.xlim(0, max(draw_range))
plt.grid(True)
plt.xlabel('The number of retrieved audio')
plt.ylabel('precision')
plt.legend()
plt.savefig("$path$/log/name.png")
plt.show()
