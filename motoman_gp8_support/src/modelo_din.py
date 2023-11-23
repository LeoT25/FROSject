#!/usr/bin/env python3

import rbdl
import numpy as np


# Lectura del modelo del robot a partir de URDF (parsing)
modelo = rbdl.loadModel('../urdf/gp8_modelo.urdf')
# Grados de libertad
ndof = modelo.q_size

# Configuracion articular
q = np.array([0.2, 1.4, 0.2, 1.3, 0.8, 2.1, 0.6])
# Velocidad articular
dq = np.array([0.1,0.8, 0.7, 0.8, 0.6, 0.9, 1.0])
# Aceleracion articular
ddq = np.array([0.01,0.2, 0.5, 0.4, 0.3, 1.0, 0.5])

# Arrays numpy
zeros = np.zeros(ndof)          # Vector de ceros
tau   = np.zeros(ndof)          # Para torque
g     = np.zeros(ndof)          # Para la gravedad
c     = np.zeros(ndof)          # Para el vector de Coriolis+centrifuga
M     = np.zeros([ndof, ndof])  # Para la matriz de inercia
e     = np.eye(ndof)               # Vector identidad

# Torque dada la configuracion del robot
rbdl.InverseDynamics(modelo, q, dq, ddq, tau)

# Calcula vector de gravedad, vector de Coriolis/centrifuga,
# y matriz M usando solamente InverseDynamics
rbdl.InverseDynamics(modelo, q, zeros, zeros, g)
print("g:\n", g.round(3))

aux   = np.zeros(ndof) 
rbdl.InverseDynamics(modelo, q, dq, zeros, aux)
c = aux-g
print("\nc:\n",c.round(3))

for i in range(ndof):
  rbdl.InverseDynamics(modelo, q, zeros, e[i,0:ndof], aux)
  M[i,0:ndof] = aux - g

M = M.T
print("\nM:\n",np.round((np.ndarray.tolist(M)),3))

# Calcula M y los efectos no lineales b
b2 = np.zeros(ndof)          # Para efectos no lineales
M2 = np.zeros([ndof, ndof])  # Para matriz de inercia

rbdl.CompositeRigidBodyAlgorithm(modelo, q, M2)

print("\nM2:\n",M2.round(3))

rbdl.NonlinearEffects(modelo,q,dq,b2)

print("\nb2:\n",b2.round(3))

# Verificacion de la expresion de la dinamica
tau_test = M@ddq+c+g
tau_test = tau_test.T

print("\nTau calculado explícitamente:\n",tau_test.round(3))
print("\nTau con InverseDynamics:\n",tau.round(3))
