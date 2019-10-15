import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

def project_and_scale(p):
    p = p[0:2,:]
    b = np.sum(p,axis=0)
    coeff = np.max(b)
    norm = np.linalg.norm(p,axis=0)
    norm[np.where(norm==0)]=1
    t = (1/coeff)*b*p/norm
    return t,coeff

def stereo_projection(t):
    return np.array([2*t[0,:]/(1+t[0,:]**2+t[1,:]**2),\
                     2*t[1,:]/(1+t[0,:]**2+t[1,:]**2),\
                     (1-t[0,:]**2-t[1,:]**2)/(1+t[0,:]**2+t[1,:]**2)])

def scale_ellipse(p,coeff,height):
    #A = np.array([[np.sqrt(3),0,0],[0,np.sqrt(3),0],[0,0,2]])
    A = np.array([[coeff,0,0],[0,coeff,0],[0,0,height]])
    return A@p+np.array([[0],[0],[1]])


x = np.linspace(0,1.5,10)
y = np.linspace(0,1.5,10)
X,Y = np.meshgrid(x,y)
height = 0.5
Z = 1+height-2*(X+Y)/3

p = np.array([X.flatten(),Y.flatten(),Z.flatten()])
v =np.array(p[2,:]>=1)
p = p[:,v]
fig = plt.figure()
ax  = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(p[0,:],p[1,:],p[2,:],'b')

q,coeff = project_and_scale(p)
q = stereo_projection(q)
q = scale_ellipse(q,coeff,height)
ax.scatter(q[0,:],q[1,:],q[2,:],'r',marker='v')
plt.show()



