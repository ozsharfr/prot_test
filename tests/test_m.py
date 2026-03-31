import numpy as np
import prody

prody.confProDy(verbosity='none')

pdb_path = "data/structures/1a22.pdb"
structure = prody.parsePDB(pdb_path, model=1)
calpha = structure.select("calpha and protein")

print("coords dtype:", calpha.getCoords().dtype)

coords = calpha.getCoords().astype("float64")
calpha.setCoords(coords)

anm = prody.ANM("test")
anm.buildHessian(calpha)
anm.calcModes(n_modes=5)

print("eigvecs dtype:", anm.getEigvecs().dtype)
print("eigvals dtype:", anm.getEigvals().dtype)

# Try getting eigvecs manually
eigvecs = np.array(anm.getEigvecs(), dtype='float64')
eigvals = np.array(anm.getEigvals(), dtype='float64')
print("cast eigvecs dtype:", eigvecs.dtype)
print("cast eigvals dtype:", eigvals.dtype)

n_atoms = calpha.numAtoms()
msf = np.zeros(n_atoms, dtype='float64')
for m in range(eigvecs.shape[1]):
    if eigvals[m] > 1e-6:
        v = eigvecs[:, m].reshape(n_atoms, 3)
        msf += np.sum(v**2, axis=1) / eigvals[m]

print("MSF computed successfully, shape:", msf.shape)
print("First 5 values:", msf[:5])

# Add this right after "MSF computed successfully"
import prody
msf2 = prody.calcSqFlucts(anm)
print("calcSqFlucts dtype:", msf2.dtype)
print("calcSqFlucts values:", msf2[:5])