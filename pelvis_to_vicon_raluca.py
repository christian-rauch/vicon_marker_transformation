import numpy as np
import vtk


part = "pelvis"  # "palm", "pelvis"
#part = "palm"  # "palm", "pelvis"

if part is "pelvis":
    # Used for experiments - 3 May 2016
    # from pelvis_val.vsk
    m1 = np.array([42.430198669433594, 0.38291072845458984, -75.745498657226562]) * 0.001
    m2 = np.array([36.803276062011719, 0.23376750946044922, -16.981033325195312]) * 0.001
    m3 = np.array([-24.306129455566406, -119.755126953125, 47.446334838867188]) * 0.001
    m4 = np.array([-54.927345275878906, 119.13845062255859, 45.280197143554688]) * 0.001

    # measurment info from jlack:
    point1 = np.array([0.1099,0,-0.2015])
    point2 = np.array([0.1099,0,-0.14053])
    point3 = np.array([0.0512,-0.1194,-0.0717])
    point4 = np.array([0.0512,0.1194,-0.0717])
    # 5cm and 2.5cm extensions with 7.5mm diameter marker dots
    point1[0] += 0.0575
    point2[0] += 0.0575
    point3[0] += 0.0575
    point4[0] += 0.0325

if part is "palm":
    m1 = np.array([8.1811103820800781, -46.510673522949219, -9.489501953125]) * 0.001
    m2 = np.array([2.6015510559082031, -14.017784118652344, 14.08099365234375]) * 0.001
    m3 = np.array([-5.1872062683105469, 32.794532775878906, -37.7120361328125]) * 0.001
    m4 = np.array([-5.5954551696777344, 27.733924865722656, 33.12054443359375]) * 0.001

    # measurment info from jlack:
    point1 = np.array([-0.0242, 0.103172, 0.02345])
    point2 = np.array([-0.0255, 0.0629, 0.02345])
    point3 = np.array([-0.025, 0.054, -0.046])
    point4 = np.array([-0.0255, 0.0173, 0.0143])
    # 5cm and 2.5cm extensions with 7.5mm diameter marker dots
    point1[0] += -0.035
    point2[0] += -0.035
    point3[0] += -0.035
    point4[0] += -0.035

#store points as vertical vectors
p = np.matrix([point1, point2, point3, point4]).transpose()
m = np.matrix([m1, m2, m3, m4]).transpose()

pointsNo = p.shape[1]

pMean = np.empty([3,1])
mMean = np.empty([3,1])

for col in range(pointsNo):
    mMean += m[:, col]
    pMean += p[:, col]

mMean /= pointsNo
pMean /= pointsNo

w = np.empty([3,3])

for col in range(pointsNo):
    w = w + (p[:, col] - pMean) * (m[:, col] - mMean).transpose()

[u,s,vT] = np.linalg.svd(w)

r = u * vT
t = pMean - r * mMean

pelvisToVicon = vtk.vtkMatrix4x4()

for row in range(3):
    for col in range(3):
        pelvisToVicon.SetElement(row, col, r[row, col])

for row in range(3):
    pelvisToVicon.SetElement(row, 3, t[row, 0])

vtkPelvisToVicon = vtk.vtkTransform()
vtkPelvisToVicon.SetMatrix(pelvisToVicon)


print vtkPelvisToVicon

