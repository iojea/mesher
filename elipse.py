import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

def affine_transf(p):
    #M = np.array([[np.sqrt(3)*3/2,0,-np.sqrt(3)],[0,np.sqrt(3)*3/2,-np.sqrt(3)],[np.sqrt(3),np.sqrt(3),3*np.sqrt(3)/2]])
    M = np.array([[1.5,0,np.sqrt(39/8)],\
        [0,1.5,np.sqrt(39/8)],\
        [-1,-1,1]])
    #M = M/np.linalg.det(M)
    M = np.linalg.inv(M)
    q = M@(p-np.array([[0],[0],[2]]))
    return q

def project_and_scale(p):
    p = p[0:2,:]
    t = p/np.linalg.norm(p,axis=0)
    return t

def stereo_projection(t):
    return np.array([2*t[0,:]/(1+t[0,:]**2+t[1,:]**2),\
                     2*t[1,:]/(1+t[0,:]**2+t[1,:]**2),\
                     (1-t[0,:]**2-t[1,:]**2)/(1+t[0,:]**2+t[1,:]**2)])

def scale_ellipse(p):
    #A = np.array([[np.sqrt(3),0,0],[0,np.sqrt(3),0],[0,0,2]])
    A = np.array([[1.5,0,0],[0,1.5,0],[0,0,1]])
    return A@p+np.array([[0],[0],[1]])


x = np.linspace(0,1.5,50)
y = np.linspace(0,1.5,50)
X,Y = np.meshgrid(x,y)
Z = 2-2*(X+Y)/3

p = np.array([X.flatten(),Y.flatten(),Z.flatten()])
v =np.array(p[2,:]>=1)
p = p[:,v]
fig = plt.figure()
ax  = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(p[0,:],p[1,:],p[2,:],'b')

q = project_and_scale(p)
print(q.shape)
print(np.linalg.norm(q))
q = scale_ellipse(stereo_projection(q))
ax.scatter(q[0,:],q[1,:],q[2,:],'r')

plt.show()

#p = np.array([x,2-2*x/3])
#q = affine_transf(p)
#q = scale_ellipse(stereo_projection(q[0,:]))
#
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot(p[0,:],p[1,:],'b')
#ax.plot(q[0,:],q[1,:],'r')
#
#x = np.linspace(0,1.5,100)
#p = np.array([x,1.5-2*x/3])
#q = affine_transf(p)
#q = scale_ellipse(stereo_projection(q[0,:]))


#ax.plot(p[0,:],p[1,:],'g')
#ax.plot(q[0,:],q[1,:],'k')
#plt.show()

