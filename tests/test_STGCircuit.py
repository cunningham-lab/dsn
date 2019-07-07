import tensorflow as tf
import numpy as np
import scipy
from tf_util.stat_util import approx_equal
from dsn.util.systems import system, Linear2D, STGCircuit, V1Circuit, SCCircuit, LowRankRNN
import matplotlib.pyplot as plt
#import dsn.lib.LowRank.Fig1_Spontaneous.fct_mf as mf

DTYPE = tf.float64
EPS = 1e-16

class stg_circuit:
    def __init__(self, dt=0.025, T=2400, fft_start=400, w=40):
        self.dt = dt
        self.T = T
        self.fft_start = fft_start
        self.w = w
        """V_m0 = -65.0e-3*np.ones((5,))
        N_0 = 0.25*np.ones((5,))
        H_0 = 0.1*np.ones((5,))
        self.init_conds = np.concatenate((V_m0, N_0, H_0), axis=0)"""
        self.init_conds = np.array([-0.04169771, -0.04319491,  0.00883992, -0.06879824,  0.03048103,
                                     0.00151316, 0.19784773, 0.56514935, 0.12214069, 0.35290397,
                                     0.08614699, 0.04938177, 0.05568701, 0.07007949, 0.05790969])

    def simulate(self, g_el, g_synA, g_synB):
        # define fixed parameters

        #conductances
        C_m = 1.0e-9

        # volatages
        V_leak = -40.0e-3 # 40 mV
        V_Ca = 100.0e-3 # 100mV
        V_k = -80.0e-3 # -80mV
        V_h = -20.0e-3 # -20mV
        V_syn = -75.0e-3 # -75mV

        v_1 = 0.0 # 0mV
        v_2 = 20.0e-3 # 20mV
        v_3 = 0.0 # 0mV
        v_4 = 15.0e-3 # 15mV
        v_5 = 78.3e-3 # 78.3mV
        v_6 = 10.5e-3 # 10.5mV
        v_7 = -42.2e-3 # -42.2mV
        v_8 = 87.3e-3 # 87.3mV
        v_9 = 5.0e-3  # 5.0mV

        v_th = -25.0e-3 # -25mV

        # neuron specific conductances
        g_Ca_f = 1.9e-2 * (1e-6) # 1.9e-2 \mu S
        g_Ca_h = 1.7e-2 * (1e-6) # 1.7e-2 \mu S
        g_Ca_s = 8.5e-3 * (1e-6) # 8.5e-3 \mu S

        g_k_f  = 3.9e-2 * (1e-6) # 3.9e-2 \mu S
        g_k_h  = 1.9e-2 * (1e-6) # 1.9e-2 \mu S
        g_k_s  = 1.5e-2 * (1e-6) # 1.5e-2 \mu S

        g_h_f  = 2.5e-2 * (1e-6) # 2.5e-2 \mu S
        g_h_h  = 8.0e-3 * (1e-6) # 8.0e-3 \mu S
        g_h_s  = 1.0e-2 * (1e-6) # 1.0e-2 \mu S

        g_Ca = np.array([g_Ca_f, g_Ca_f, g_Ca_h, g_Ca_s, g_Ca_s])
        g_k = np.array([g_k_f, g_k_f, g_k_h, g_k_s, g_k_s])
        g_h = np.array([g_h_f, g_h_f, g_h_h, g_h_s, g_h_s])

        g_leak = 1.0e-4 * (1e-6) # 1e-4 \mu S

        phi_N = 2 # 0.002 ms^-1

        def f(x, g_el, g_synA, g_synB):
            # x contains
            V_m = x[:5]
            N = x[5:10]
            H = x[10:]
            
            M_inf = 0.5*(1.0 + np.tanh((V_m - v_1)/ v_2))
            N_inf = 0.5*(1.0 + np.tanh((V_m - v_3)/v_4))
            H_inf = 1.0 / (1.0 + np.exp((V_m + v_5)/v_6))
                           
            S_inf = 1.0 / (1.0 + np.exp((v_th - V_m) / v_9))
            
            I_leak = g_leak*(V_m - V_leak)
            I_Ca = g_Ca*M_inf*(V_m - V_Ca)
            I_k = g_k*N*(V_m - V_k)
            I_h = g_h*H*(V_m - V_h)
                           
            I_elec = np.array([0.0, 
                               g_el*(V_m[1]-V_m[2]),
                               g_el*(V_m[2]-V_m[1] + V_m[2]-V_m[4]),
                               0.0,
                               g_el*(V_m[4]-V_m[2])])
                           
            I_syn = np.array([g_synB*S_inf[1]*(V_m[0] - V_syn),
                                g_synB*S_inf[0]*(V_m[1] - V_syn),
                                g_synA*S_inf[0]*(V_m[2] - V_syn) + g_synA*S_inf[3]*(V_m[2] - V_syn),
                                g_synB*S_inf[4]*(V_m[3] - V_syn),
                                g_synB*S_inf[3]*(V_m[4] - V_syn)])

            I_total = I_leak + I_Ca + I_k + I_h + I_elec + I_syn    
            
            lambda_N = (phi_N)*np.cosh((V_m - v_3)/(2*v_4))
            tau_h = (272.0 - (-1499.0 / (1.0 + np.exp((-V_m + v_7) / v_8)))) / 1000.0
            
            dVmdt = (1.0 / C_m)*(-I_total)
            dNdt = lambda_N*(N_inf - N)
            dHdt = (H_inf - H) / tau_h
            
            dxdt = np.concatenate((dVmdt, dNdt, dHdt), axis=0)
            return dxdt

        x = self.init_conds
        xs = [x]
        for t in range(self.T):
            dxdt = f(x, g_el, g_synA, g_synB)
            x = dxdt*self.dt + x
            xs.append(x)
        X = np.array(xs)

        return X

    def T_x(self, g_el, g_synA, g_synB):
        def moving_average(a, n=3) :
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1:] / n

        # sampling frequency
        Fs  = 1.0 / self.dt 
        # num samples for freq measurement
        N = self.T - self.fft_start + 1 - (self.w-1) 
        
        min_freq = 0.0
        max_freq = 1.0
        num_freqs = 101
        freqs = np.linspace(min_freq, max_freq, num_freqs)

        ns = np.arange(0,N)
        phis = []
        for i in range(num_freqs):
            k = N*freqs[i] / Fs
            phi = np.cos(2*np.pi*k*ns/N) - 1j * np.sin(2*np.pi*k*ns/N)
            phis.append(phi)

        Phi = np.array(phis)

        alpha = 100

        X = self.simulate(g_el, g_synA, g_synB)
        v_h = X[self.fft_start:,2]

        v_h_rect = np.maximum(v_h, -0.01)
        v_h_rect_LPF = moving_average(v_h_rect, self.w)
        #v_h_rect_LPF = v_h_rect_LPF - np.mean(v_h_rect_LPF)
        
        V_h = np.abs(np.dot(Phi, v_h_rect_LPF))
        V_h_pow = np.power(V_h, alpha)
        freq_id = V_h_pow / np.sum(V_h_pow)

        freq = np.dot(freqs, freq_id)
        T_x = np.array([freq, np.square(freq)])
        return T_x

def test_STGCircuit():
    np.random.seed(0)
    M = 100 

    dt = 0.025
    T = 200
    fft_start = 0
    w = 20

    true_sys = stg_circuit(dt, T, fft_start, w=w)

    mean = 0.55
    variance = 0.0001
    fixed_params = {}
    behavior = {"type":"hubfreq",
                "mean":mean,
                "variance":variance}
    model_opts = {"dt":dt,
                  "T":T,
                  "fft_start":fft_start,
                  "w":w
                 }
    system = STGCircuit(fixed_params, behavior, model_opts)
    assert system.name == "STGCircuit"
    assert system.behavior_str == "hubfreq_mu=5.50E-01_3.03E-01"
    assert approx_equal(system.behavior["mean"], 0.55, EPS)
    assert system.all_params == [
        "g_el",
        "g_synA",
        "g_synB",
    ]
    assert system.free_params == [
        "g_el",
        "g_synA",
        "g_synB"
    ]
    assert system.z_labels == [
        r"$g_{el}$",
        r"$g_{synA}$",
        r"$g_{synB}$"
    ]
    assert system.T_x_labels == [
        r"$f_{h}$",
        r"$f_{h}^2$"
    ]
    assert system.D == 3
    assert system.num_suff_stats == 2

    Z = tf.placeholder(dtype=DTYPE, shape=(1,M,3))
    _Z = np.random.uniform(0.0, 20.0, (1,M,2)) * 1e-9
    synB = 5e-9
    _Z = np.concatenate((_Z, synB*np.ones((1,M, 1))), axis=2)
    x_true = np.zeros((M,15,T+1))
    T_x_true = np.zeros((M,2))
    for i in range(M):
        g_el = _Z[0,i,0]
        g_synA = _Z[0,i,1]
        g_synB = _Z[0,i,2]
        x_true[i,:,:] = true_sys.simulate(g_el, g_synA, g_synB).T
        T_x_true[i,:] = true_sys.T_x(g_el, g_synA, g_synB)

    T_x = system.compute_suff_stats(Z)
    x_t = system.simulate(Z, db=True)
    with tf.Session() as sess:
        _x_t, _T_x = sess.run([x_t, T_x], {Z:_Z/1.0e-9})

    assert(approx_equal(np.transpose(_x_t, [1,2,0]), x_true, EPS))
    assert(approx_equal(_T_x[0], T_x_true, EPS, allow_special=True))

    return None

if __name__ == "__main__":
    test_STGCircuit()
