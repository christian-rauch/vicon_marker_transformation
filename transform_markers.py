#!usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# <Parameter NAME="val_palm_left_val_palm_left1_x" VALUE="8.1811103820800781" PRIOR="8.1811103820800781" HIDDEN="1"/>
# <Parameter NAME="val_palm_left_val_palm_left1_y" VALUE="-46.510673522949219" PRIOR="-46.510673522949219" HIDDEN="1"/>
# <Parameter NAME="val_palm_left_val_palm_left1_z" VALUE="-9.489501953125" PRIOR="-9.489501953125" HIDDEN="1"/>
# <Parameter NAME="val_palm_left_val_palm_left2_x" VALUE="2.6015510559082031" PRIOR="2.6015510559082031" HIDDEN="1"/>
# <Parameter NAME="val_palm_left_val_palm_left2_y" VALUE="-14.017784118652344" PRIOR="-14.017784118652344" HIDDEN="1"/>
# <Parameter NAME="val_palm_left_val_palm_left2_z" VALUE="14.08099365234375" PRIOR="14.08099365234375" HIDDEN="1"/>
# <Parameter NAME="val_palm_left_val_palm_left3_x" VALUE="-5.1872062683105469" PRIOR="-5.1872062683105469" HIDDEN="1"/>
# <Parameter NAME="val_palm_left_val_palm_left3_y" VALUE="32.794532775878906" PRIOR="32.794532775878906" HIDDEN="1"/>
# <Parameter NAME="val_palm_left_val_palm_left3_z" VALUE="-37.7120361328125" PRIOR="-37.7120361328125" HIDDEN="1"/>
# <Parameter NAME="val_palm_left_val_palm_left4_x" VALUE="-5.5954551696777344" PRIOR="-5.5954551696777344" HIDDEN="1"/>
# <Parameter NAME="val_palm_left_val_palm_left4_y" VALUE="27.733924865722656" PRIOR="27.733924865722656" HIDDEN="1"/>
# <Parameter NAME="val_palm_left_val_palm_left4_z" VALUE="33.12054443359375" PRIOR="33.12054443359375" HIDDEN="1"/>

part = "pelvis"  # "palm", "pelvis"
#part = "palm"  # "palm", "pelvis"

if part is "palm":
    marker = np.matrix([[
    8.1811103820800781, -46.510673522949219, -9.489501953125
    ], [
    2.6015510559082031, -14.017784118652344, 14.08099365234375
    ], [
    -5.1872062683105469, 32.794532775878906, -37.7120361328125
    ], [
    -5.5954551696777344, 27.733924865722656, 33.12054443359375
    ]])

    # from Vicon marker position in mm to SI unit meter
    marker *= 0.001


    hand = np.matrix([[
    -0.0242, 0.103172, 0.02345
    ], [
    -0.0255, 0.0629, 0.02345
    ], [
    -0.025, 0.054, -0.046
    ], [
    -0.0255, 0.0173, 0.0143
    ]])

    hand[:,0] += -0.035


# # pelvis marker
if part is "pelvis":
    marker = np.matrix([[
        42.430198669433594, 0.38291072845458984, -75.745498657226562
    ], [
        36.803276062011719, 0.23376750946044922, -16.981033325195312
    ], [
        -24.306129455566406, -119.755126953125, 47.446334838867188
    ], [
        -54.927345275878906, 119.13845062255859, 45.280197143554688
    ]]) * 0.001

    hand = np.matrix([[0.1099, 0, -0.2015],
    [0.1099, 0, -0.14053],
    [0.0512, -0.1194, -0.0717],
    [0.0512, 0.1194, -0.0717]])

    offset = np.matrix([0.0575, 0.0575, 0.0575, 0.0325]).T
    hand[:,0] += offset


# centre data
m_mean = np.mean(marker, axis=0)
p_mean = np.mean(hand, axis=0)

print "marker:\n",marker
print "maker mean", m_mean
print "hand points:\n",hand
print "hand mean", p_mean

m_norm = np.subtract(marker, m_mean)
p_norm = np.subtract(hand, p_mean)
#print m_norm
#print p_norm

m_norm_mean = np.mean(m_norm, axis=0)
p_norm_mean = np.mean(p_norm, axis=0)
print "maker n mean", m_norm_mean
print "hand n mean", p_norm_mean


# get transformation marker to hand frame
# http://soe.rutgers.edu/~meer/TEACH/ADD/ls3duma.pdf

# covariance matrix with centred data
X = m_norm.T * p_norm

print "det X:",np.linalg.det(X)
print "rank X:",np.linalg.matrix_rank(X)

#print "X",X

# rotation from SVD
[U,S,Vt] = np.linalg.svd(X)

print "det U, det V",np.linalg.det(U),np.linalg.det(U)

#print U,V
R = Vt.T*U.T

print "R",R
print "det R:",np.linalg.det(R)
print "det X:",np.linalg.det(X)
print "rank X:",np.linalg.matrix_rank(X)

# translation from rotated mean
t = p_mean - (R * m_mean.T).T

print "t",t

T = np.zeros((4,4))
T[0:3,0:3] = R
T[0:3,3] = t
T[3,3] = 1

# transformation marker to frame
print "T",T

# transformation frame to marker, e.g. the pose of the robot frame in the world frame
print "T^-1",np.linalg.inv(T)

# project back for visual test
hand2 = t + (R * marker.T).T
print hand2

err = np.mean(hand - hand2, axis=0)
print "backprojection err", np.linalg.norm(err)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.axis('equal')

# show vicon marker in marker frame
ax.scatter(0, 0, 0, c='black')
ax.scatter(marker[0, 0], marker[0, 1], marker[0, 2], c='r')
ax.scatter(marker[1, 0], marker[1, 1], marker[1, 2], c='g')
ax.scatter(marker[2, 0], marker[2, 1], marker[2, 2], c='b')
ax.scatter(marker[3, 0], marker[3, 1], marker[3, 2], c='y')

# show vicon marker in robot frame
ax.scatter(hand[0, 0], hand[0, 1], hand[0, 2], c='r', marker='s')
ax.scatter(hand[1, 0], hand[1, 1], hand[1, 2], c='g', marker='s')
ax.scatter(hand[2, 0], hand[2, 1], hand[2, 2], c='b', marker='s')
ax.scatter(hand[3, 0], hand[3, 1], hand[3, 2], c='y', marker='s')
#ax.scatter(-0.0242, 0.094, -0.0509, c='c', marker='x')

# show projection of vicon marker points into robot frame
ax.scatter(hand2[0, 0], hand2[0, 1], hand2[0, 2], c='r', marker='x', s=100)
ax.scatter(hand2[1, 0], hand2[1, 1], hand2[1, 2], c='g', marker='x', s=100)
ax.scatter(hand2[2, 0], hand2[2, 1], hand2[2, 2], c='b', marker='x', s=100)
ax.scatter(hand2[3, 0], hand2[3, 1], hand2[3, 2], c='y', marker='x', s=100)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

plt.show()