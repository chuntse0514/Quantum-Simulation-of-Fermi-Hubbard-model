import openfermion as of
from openfermion import MolecularData
from openfermionpyscf import run_pyscf

def H2(r, basis='sto-3g', multiplicity=1, charge=0) -> MolecularData:
    geometry = [['H', [0., 0., 0.]], ['H', [0., 0., r]]]
    h2_molecule = MolecularData(geometry, basis, multiplicity, charge)
    h2_molecule = run_pyscf(h2_molecule, run_ccsd=True, run_fci=True)

    # uncomment to get access to the hf, ccsd and fci energy
    # print(h2_molecule.hf_energy)
    # print(h2_molecule.ccsd_energy)
    # print(h2_molecule.fci_energy)
    return h2_molecule

def HeH_Ion(r, basis='sto-3g', multiplicity=1, charge=1) -> MolecularData:
    geometry = [['He', [0., 0., 0.]],['H', [0., 0., r]]]
    heh_ion = MolecularData(geometry, basis, multiplicity, charge)
    heh_ion = run_pyscf(heh_ion, run_ccsd=True, run_fci=True)
    return heh_ion

def LiH(r, basis='sto-3g', multiplicity=1, charge=0) -> MolecularData:
    geometry = [['Li', [0., 0., 0.]], ['H', [0., 0., r]]]
    lih_molecule = MolecularData(geometry, basis, multiplicity, charge)
    lih_molecule = run_pyscf(lih_molecule, run_ccsd=True, run_fci=True)
    return lih_molecule
 
def BeH2(r, basis='sto-3g', multiplicity=1, charge=0) -> MolecularData:
    geometry = [['H', [0., 0., -r]], ['Be', [0., 0., 0.]], ['H', [0., 0., r]]]
    beh2_molecule = MolecularData(geometry, basis, multiplicity, charge)
    beh2_molecule = run_pyscf(beh2_molecule, run_ccsd=True, run_fci=True) 
    return beh2_molecule

def H4(r, basis='sto-3g', multiplicity=1, charge=0) -> MolecularData:
    geometry = [['H', [0., 0., 0.]], ['H', [0., 0., r]], ['H', [0., 0., 2*r]], ['H', [0., 0., 3*r]]]
    h4_molecule = MolecularData(geometry, basis, multiplicity, charge)
    h4_molecule = run_pyscf(h4_molecule, run_ccsd=True, run_fci=True)
    return h4_molecule

def H6(r, basis='sto-3g', multiplicity=1, charge=0) -> MolecularData:
    geometry = [['H', [0., 0., 0.]], ['H', [0., 0., r]], ['H', [0., 0., 2*r]],
                ['H', [0., 0., 3*r]], ['H', [0., 0., 4*r]], ['H', [0., 0., 5*r]]]
    h6_molecule = MolecularData(geometry, basis, multiplicity, charge)
    h6_molecule = run_pyscf(h6_molecule, run_ccsd=True, run_fci=True)
    return h6_molecule

def test_molecule():
    molecule = LiH(r=2)
    num_qubit = molecule.n_qubits
    num_electron = molecule.n_electrons
    num_orbital = molecule.n_orbitals

    # print(num_qubit)
    # print(num_electron)   
    # print(num_orbital)
    # print(molecule.canonical_orbitals)
    # print(molecule.one_body_integrals)
    # print(molecule.two_body_integrals)

    # print(of.get_sparse_operator(molecule.get_molecular_hamiltonian()))