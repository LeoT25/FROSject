import numpy as np
from copy import copy
import rbdl

pi = np.pi

class Robot(object):
    def __init__(self, q0, dq0, ndof, dt):
        self.q = q0    # numpy array (ndof x 1)
        self.dq = dq0  # numpy array (ndof x 1)
        self.M = np.zeros([ndof, ndof])
        self.b = np.zeros(ndof)
        self.dt = dt
        self.robot = rbdl.loadModel('../urdf/gp8_modelo.urdf')

    def send_command(self, tau):
        rbdl.CompositeRigidBodyAlgorithm(self.robot, self.q, self.M)
        rbdl.NonlinearEffects(self.robot, self.q, self.dq, self.b)
        ddq = np.linalg.inv(self.M).dot(tau-self.b)
        self.q = self.q + self.dt*self.dq
        self.dq = self.dq + self.dt*ddq

        # -----------------------------
        # Limites
        if(self.q[0]<-0.5000): self.q[0]=-0.5000
        if(self.q[0]> 0.5000): self.q[0]=0.5000

        if(self.q[1]<-2.9670): self.q[1]=-2.9670
        if(self.q[1]> 2.9670): self.q[1]=2.9670

        if(self.q[2]<-1.1344): self.q[2]=-1.1344
        if(self.q[2]> 2.5307): self.q[2]=2.5307

        if(self.q[3]<-1.2217): self.q[3]=-1.2217
        if(self.q[3]> 3.3161): self.q[3]=3.3161

        if(self.q[4]<-3.3161): self.q[4]=-3.3161
        if(self.q[4]> 3.3161): self.q[4]=3.3161

        if(self.q[5]<-2.3561): self.q[5]=-2.3561
        if(self.q[5]> 2.3561): self.q[5]=2.3561

        if(self.q[6]<-6.2944): self.q[6]=-6.2944
        if(self.q[6]> 6.2944): self.q[6]=6.2944

        # -----------------------------
        # Limites
        if(self.dq[0]<-0.02): self.dq[0]=-0.02
        if(self.dq[0]> 0.02): self.dq[0]=0.02

        if(self.dq[1]<-7.9412): self.dq[1]=-7.9412
        if(self.dq[1]> 7.9412): self.dq[1]=7.9412

        if(self.dq[2]<-6.7495): self.dq[2]=-6.7495
        if(self.dq[2]> 6.7495): self.dq[2]=6.7495

        if(self.dq[3]<-9.0757): self.dq[3]=-9.0757
        if(self.dq[3]> 9.0757): self.dq[3]=9.0757

        if(self.dq[4]<-9.5993): self.dq[4]=-9.5993
        if(self.dq[4]> 9.5993): self.dq[4]=9.5993

        if(self.dq[5]<-9.5993): self.dq[5]=-9.5993
        if(self.dq[5]> 9.5993): self.dq[5]=9.5993

        if(self.dq[6]<-17.4845): self.dq[6]=-17.4845
        if(self.dq[6]> 17.4845): self.dq[6]=17.4845

    def read_joint_positions(self):
        return self.q

    def read_joint_velocities(self):
        return self.dq

def dh(d, theta, a, alpha):
    """
    Calcular la matriz de transformacion homogenea asociada con los parametros
    de Denavit-Hartenberg.
    Los valores d, theta, a, alpha son escalares.

    """
    sth = np.sin(theta)
    cth = np.cos(theta)
    sa  = np.sin(alpha)
    ca  = np.cos(alpha)
    T = np.array([[cth, -ca*sth,  sa*sth, a*cth],
                  [sth,  ca*cth, -sa*cth, a*sth],
                  [0.0,      sa,      ca,     d],
                  [0.0,     0.0,     0.0,   1.0]])
    return T


def fkine_mmgp8(q):
    """
    Calcular la cinematica directa del robot UR5 dados sus valores articulares. 
    q es un vector numpy de la forma [q1, q2, q3, q4, q5, q6]

    """

    # Matrices DH (completar)
    T0 = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
    T1 = dh(q[0],0,0,pi/2)
    T2 = dh(0.330+0.076+0.01, q[1], 0.04, pi/2)
    T3 = dh(0,-q[2]+pi/2, 0.345, 0)
    T4 = dh(0,q[3],0.04,pi/2)
    T5 = dh(0.340,-q[4],0,-pi/2)
    T6 = dh(0,q[5],0,pi/2)
    T7 = dh(0.08,q[6],0,0)
    # Efector final con respecto a la base
    T = T0@T1@T2@T3@T4@T5@T6@T7
    return T


def jacobian_position(q, delta=0.0001):
    """
    Jacobiano analitico para la posicion. Retorna una matriz de 3x6 y toma como
    entrada el vector de configuracion articular q=[q1, q2, q3, q4, q5, q6, q7]
    """
    # Alocacion de memoria
    J = np.zeros((3,7))
    # Transformacion homogenea inicial (usando q)
    T = fkine_mmgp8(q)
    # Iteracion para la derivada de cada columna
    for i in range(7):
        # Copiar la configuracion articular inicial (usar este dq para cada
        # incremento en una articulacion)
        dq = copy(q)
        # Calcular nuevamenta la transformacion homogenea e
        # Incrementar la articulacion i-esima usando un delta
        dq[i] = dq[i] + delta
        # Transformacion homogenea luego del incremento (q+delta)
        T_inc = fkine_mmgp8(dq)
        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        J[0:3,i]=(T_inc[0:3,3]-T[0:3,3])/delta
    return J


def ikine_mmgp8(xdes, q0):
    """
    Calcular la cinematica inversa de UR5 numericamente a partir de la configuracion articular inicial de q0.
    Emplear el metodo de newton
    """
    epsilon  = 0.001
    max_iter = 1000
    # Files for the logs
    error = open("/home/user/lab_ws/errorac.txt", "w")    

    q  = copy(q0)
    for i in range(max_iter):
        # Main loop
        J = jacobian_position(q)
        f = fkine_mmgp8(q)
        e = xdes - f[0:3,3]
        q = q + np.dot(J.T, e)
        norm_e = np.linalg.norm(e)
        error.write(str(e[0])+' '+str(e[1]) +' '+str(e[2])+' '+str(norm_e)+'\n')
        if (norm_e < epsilon):
            break
    return q

def ik_gradient_mmgp8(xdes, q0):
    """
    Calcular la cinematica inversa de UR5 numericamente a partir de la configuracion articular inicial de q0.
    Emplear el metodo gradiente
    """
    epsilon  = 0.001
    max_iter = 1000
    alpha = 0.1

    q  = copy(q0)
    for i in range(max_iter):
        # Main loop
        J = jacobian_position(q)
        f = fkine_mmgp8(q)
        e = xdes - f[0:3,3]
        q = q + alpha*np.dot(J.T, e)

        if (np.linalg.norm(e) < epsilon):
            break  
    return q
