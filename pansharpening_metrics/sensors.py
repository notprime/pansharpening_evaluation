import numpy as np

SENSOR_MTF = {
    'QB': {
        'GNyq': np.array([0.34, 0.32, 0.30, 0.22]),  # B, G, R, NIR
        'GNyqPan': 0.15
    },
    'IKONOS': {
        'GNyq': np.array([0.26, 0.28, 0.29, 0.28]),  # B, G, R, NIR
        'GNyqPan': 0.17
    },
    'GEOEYE1': {
        'GNyq': np.array([0.23, 0.23, 0.23, 0.23]),  # B, G, R, NIR
        'GNyqPan': 0.16
    },
    'WV2': {
        'GNyq': np.append(0.35 * np.ones(7), 0.27),  # 8 bands
        'GNyqPan': 0.11
    },
    'WV3': {
        'GNyq': np.array([0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315]),
        'GNyqPan': 0.5
    },
    'PRISMA': {
        'GNyq': 0.28,
        'GNyqPan': 0.22
    }
}