import numpy as np

quadrature_rules = {
    '16points': [np.array([[0.33333333, 0.33333333],
                           [0.45929259, 0.45929259],
                           [0.45929259, 0.08141482],
                           [0.08141482, 0.45929259],
                           [0.17056931, 0.17056931],
                           [0.17056931, 0.65886138],
                           [0.65886138, 0.17056931],
                           [0.05054723, 0.05054723],
                           [0.05054723, 0.89890554],
                           [0.89890554, 0.05054723],
                           [0.26311283, 0.72849239],
                           [0.72849239, 0.00839478],
                           [0.00839478, 0.26311283],
                           [0.72849239, 0.26311283],
                           [0.26311283, 0.00839478],
                           [0.00839478, 0.72849239]]),
                 np.array([0.14431560767779,
                           0.09509163426728,
                           0.09509163426728,
                           0.09509163426728,
                           0.10321737053472,
                           0.10321737053472,
                           0.10321737053472,
                           0.03245849762320,
                           0.03245849762320,
                           0.03245849762320,
                           0.02723031417443,
                           0.02723031417443,
                           0.02723031417443,
                           0.02723031417443,
                           0.02723031417443,
                           0.02723031417443])
                 ],
    '3points': [np.array([[1. / 6., 1. / 6.],
                          [1. / 6., 2. / 3.],
                          [2. / 3., 1. / 6.]]),
                1. / 3 * np.ones(3)
                ],
    '4points': [np.array([[1. / 3., 1. / 3.],
                          [0.2, 0.6],
                          [0.2, 0.2],
                          [0.6, 0.2]]),
                np.array([-27. / 48.,
                          25. / 48.,
                          25. / 48.,
                          25. / 48.])
                ]
}
Px = np.array([[0.33333333333333, 0.33333333333333],
               [0.47014206410511, 0.47014206410511],
               [0.47014206410511, 0.05971587178977],
               [0.05971587178977, 0.47014206410511],
               [0.10128650732346, 0.10128650732346],
               [0.10128650732346, 0.79742698535309],
               [0.79742698535309, 0.10128650732346]])
dx = 0.5 * np.array([0.22500000000000,
                     0.13239415278851,
                     0.13239415278851,
                     0.13239415278851,
                     0.12593918054483,
                     0.12593918054483,
                     0.12593918054483])
Py = Px
dy = dx
configuration = {
    'save_results': 1,

    'target_shape': 'target_shape',
    'init_shape': 'spline', # 'square',

    'c_per': 0.0002,  # 0.002,

    'boundary_label': 3,
    'interface_label': 12,

    'source': [10, -10],
    'epsilon': 0.02,
    #fractional_shape, integrable_unsym
    'kernel': dict(function="fractional_shape", horizon=1. / 10., outputdim=1, fractional_s=0.6),
    'nlfem_conf': dict(ansatz="CG",
                       approxBalls={"method": "retriangulate",
                                    "isPlacePointOnCap": True,  # required for "retriangulate" only
                                    # "averageBallWeights": [1., 1., 1.]  # required for "averageBall" only
                                    },
                       closeElements="fractional",
                       quadrature={"outer": {"points": Px,
                                             "weights": dx
                                             },
                                   "inner": {"points": Py,
                                             "weights": dy
                                             },
                                   "tensorGaussDegree": 5  # Degree of tensor Gauss quadrature for singular kernels.
                                   },
                       ShapeDerivative=0,
                       verbose=False),
    'nlfem_shape_conf': dict(ansatz="CG",  # only CG possible
                          approxBalls={"method": "retriangulate_second_shape",
                                       "isPlacePointOnCap": True,  # required for "retriangulate" only
                                        # "averageBallWeights": [1., 1., 1.]  # required for "averageBall" only
                                       },
                          closeElements="fractional_second_shape",
                          quadrature={"outer": {"points": Px,
                                                "weights": dx
                                                },
                                      "inner": {"points": Py,
                                                "weights": dy
                                                },
                                      "tensorGaussDegree": 5  # Degree of tensor Gauss quadrature for singular kernels.
                                      },
                          ShapeDerivative=1,
                          verbose=False)
}
# END
