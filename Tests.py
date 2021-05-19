from Reward import Reward
from RewardFnSpace import RewardFnSpace
from AcrobotUtils import *
from PlotUtils import *
from A2C import *
from IRL import *

def test_l1norm () : 
    n = 10
    rfs = RewardFnSpace(list(range(n)))
    for i in range(10): 
        b = rfs.bs[i]
        rfs.lp += b == 0
    rfs.lp.solve()
    rfs._setCoeffs()
    coeffs = np.array(rfs.coeffs)
    assert(np.linalg.norm(coeffs - np.ones(n)) < 1e-4)

def test_acrobotbases (): 
    xRange = np.arange(-np.pi, np.pi, 0.1)
    yRange = np.arange(-np.pi, np.pi, 0.1)
    acrobotBases = acrobotRewardBases(np.pi, np.pi)
    toExternal = lambda x, y : toExternalStateRep([x, y, 0, 0])
    for basis in acrobotBases : 
        f = compose(basis, toExternal)
        plotFunction(f, xRange, yRange, 'theta1', 'theta2', 'R')

def test_optimalagentfinder () :
    def valNetwork (s) : 
        s = s.float()
        v = reduce(model.withReluDropout, model.v[:-1], s)
        v = model.v[-1](v)
        return v
    acrobotBases = acrobotRewardBases(np.pi / 8, np.pi / 8)
    fn = random.sample(acrobotBases, k=1).pop()
    agent = findOptimalAgent(fn)
    model = agent.model
    toExternal = lambda x, y : toExternalStateRep([x, y, 0, 0])
    valFn = reduce(compose, [float, valNetwork, torch.tensor, toExternal])
    RFn = compose(fn, toExternal)
    xRange = np.arange(-np.pi, np.pi, 0.1)
    yRange = np.arange(-np.pi, np.pi, 0.1)
    plotFunction(RFn, xRange, yRange, 'theta1', 'theta2', 'R')
    plotFunction(valFn, xRange, yRange, 'theta1', 'theta2', 'V')

def test_piecewiselinearlp1 () :
    # maximize min(x, 2 * x) ; -10 <= x <= 10
    # expected answer: x = 10.
    lp = LpProblem('test', LpMaximize)
    x = LpVariable('x', -10, 10)
    t = LpVariable('t')
    lp += t
    lp += t <= x
    lp += t <= 2 * x
    lp.solve()
    optimum = value(lp.objective)
    assert(abs(optimum - 10) < 1e-3)

def test_piecewiselinearlp2 () : 
    # maximize (min(x, 2 * x) + min(3 * x, 4 * x)) 
    # -10 <= x <= 10
    # expected answer: x = 40
    lp = LpProblem('test', LpMaximize)
    x = LpVariable('x', -10, 10)
    t1 = LpVariable('t1')
    t2 = LpVariable('t2')
    lp += t1 + t2
    lp += t1 <= x
    lp += t1 <= 2 * x
    lp += t2 <= 3 * x
    lp += t2 <= 4 * x
    lp.solve()
    optimum = value(lp.objective)
    assert(abs(optimum - 40) < 1e-3)

def test_traj () :
    samples = getAllTraj()
    states = []
    for t in samples : 
        states.extend([toInternalStateRep(s) for s, _, _ in t])
    states = np.stack(states)
    xRange = np.linspace(-np.pi, np.pi, 100)
    yRange = np.linspace(-np.pi, np.pi, 100)
    plotHist(states, xRange, yRange, 'theta1', 'theta2', 'S Count')

def test_sampling1 (): 
    cpus = list(range(C.N_PARALLEL))
    affinity = dict(cuda_idx=C.CUDA_IDX, workers_cpus=cpus)
    agent_ = findOptimalAgent(reward=None)
    agent = CategoricalPgAgent(AcrobotNet, 
               initial_model_state_dict=agent_.state_dict())
    s0 = np.array([1, 0, 1/np.sqrt(2), 1/np.sqrt(2), 4, 2], dtype=np.float)
    sampler = SerialSampler(
        EnvCls=rlpyt_make,
        env_kwargs=dict(id=C.ENV, reward=None, internalStateFn=C.INTERNAL_STATE_FN, s0=s0),
        batch_T=500,
        batch_B=16,
        max_decorrelation_steps=0,
    )
    sampler.initialize(
        agent=agent,
        affinity=affinity,
        seed=0
    )
    _, traj_info = sampler.obtain_samples(0)
    print(np.mean([t['DiscountedReturn'] for t in traj_info]))

def test_sampling2 () :
    delta = 2 * np.pi / 3
    r = Reward(partial(stepFunction, 
                       xRange=(-delta/2, delta/2), 
                       yRange=(-delta/2, delta/2)), 
               (-1, 0))
    states = []
    xs = np.arange(-np.pi, np.pi, delta)
    ys = np.arange(-np.pi, np.pi, delta)
    for x, y in product(xs, ys) : 
        states.append(
            toExternalStateRep([x + delta / 2, y + delta / 2, 0, 0]).astype(float)
        )
    agent = findOptimalAgent(r)
    vals = estimateValueFromAgent(states, agent, r)
    for s, v in zip(states, vals) : 
        print(toInternalStateRep(s)[:2], v)

if __name__ == "__main__" :
    test_piecewiselinearlp1()
    test_piecewiselinearlp2()
    test_l1norm()
    test_acrobotbases()
    test_optimalagentfinder()
    test_traj()
    test_sampling2()