import matplotlib
#matplotlib.use("wxAgg")     #Remove this line to use the standard backend. "wxAgg" however runs a lot smoother on Windows. Install "wxPython" to use it.
#import matplotlib.pyplot as plt
#from matplotlib import cm, colors
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sympy as sp
from spb import plot3d_parametric_surface

uS, vS = sp.symbols("u v", real=True)

ellipsoidParam = sp.Matrix([0.6666 * (1. - 0.25 * sp.cos(sp.pi * vS)) * sp.sin(sp.pi * vS) * sp.cos(2. * sp.pi * uS), 0.6666 * (1. - 0.25 * sp.cos(sp.pi * vS)) * sp.sin(sp.pi * vS) * sp.sin(2. * sp.pi * uS), sp.cos(sp.pi * vS)]) 

manifoldParam = ellipsoidParam

manifoldJacobi = manifoldParam.jacobian([uS, vS])
firstFundamentalForm = manifoldJacobi.T @ manifoldJacobi
E = sp.simplify(firstFundamentalForm[0, 0])
F = sp.simplify(firstFundamentalForm[0, 1])
G = sp.simplify(firstFundamentalForm[1, 1])

tan0 = manifoldJacobi @ sp.Matrix([1, 0])
tan1 = manifoldJacobi @ sp.Matrix([0, 1])
manifoldNormal = tan0.cross(tan1)
manifoldNormal = manifoldNormal / manifoldNormal.norm()
manifoldUU = manifoldParam.diff(uS).diff(uS)
manifoldUV = manifoldParam.diff(uS).diff(vS)
manifoldVV = manifoldParam.diff(vS).diff(vS)
L = sp.simplify(manifoldUU.dot(manifoldNormal))
M = sp.simplify(manifoldUV.dot(manifoldNormal))
N = sp.simplify(manifoldVV.dot(manifoldNormal))

print("Computed and simplified fundamental forms.")

gaussianCurvature  = (L * N - M * M) / (E * G - F * F)
meanCurvature      = (L * G - 2. * M * F + N * E) / (2 * (E * G - F * F))

print("Computed expressions for K and M.")

gaussianCurvature = sp.simplify(gaussianCurvature)
meanCurvature = sp.simplify(meanCurvature)
geometricPotential = -0.5 * (meanCurvature * meanCurvature - gaussianCurvature) 

print("Computed expression for geometric potential.")
print("Mean curvature =", meanCurvature)
print("Gaussian curvature =", gaussianCurvature)
print("Geometric potential =", geometricPotential)

def callabaleSurfaceColor(u, v):
    #result = meanCurvature.subs(uS, u).subs(vS, v).evalf()
    #result = gaussianCurvature.subs(uS, u).subs(vS, v).evalf()
    #print(float(result), "for u =", u, ", v =", v)
    result = geometricPotential.subs(uS, u).subs(vS, v).evalf()
    return float(result)

vectorizedSurfaceColor = np.vectorize(callabaleSurfaceColor)

manifoldPlot = plot3d_parametric_surface(manifoldParam[0], manifoldParam[1], manifoldParam[2], (uS, 1e-8, 1.-1e-8), (vS, 1e-8, 1.-1e-8), use_cm=True, color_func=lambda u, v: vectorizedSurfaceColor(u, v), show=False)
manifoldPlot.aspect = "equal"
manifoldPlot.show()