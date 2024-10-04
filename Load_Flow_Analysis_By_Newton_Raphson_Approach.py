import numpy as np
import sympy as sp

def get_impedances_and_bus_types(num_buses):
    impedance_matrix = np.zeros((num_buses, num_buses), dtype=complex)
    bus_types = []

    print("Enter the impedances in the form a+bj (e.g., 1+2j) and bus types (1 for PQ, 2 for PV, 3 for slack) between the buses:")

    for i in range(num_buses):
        while True:
            try:
                bus_type = int(input(f"Bus type for bus {i+1} (1 for PQ, 2 for PV, 3 for slack): "))
                if bus_type not in [1, 2, 3]:
                    raise ValueError
                bus_types.append(bus_type)
                break
            except ValueError:
                print("Invalid input. Please enter 1, 2, or 3.")

        for j in range(i+1, num_buses):
            while True:
                try:
                    impedance = complex(input(f"Impedance between bus {i+1} and bus {j+1}: "))
                    impedance_matrix[i, j] = impedance
                    impedance_matrix[j, i] = impedance
                    break
                except ValueError:
                    print("Invalid input. Please enter a valid complex number in the form a+bj.")

    return impedance_matrix, bus_types

def get_inputs(num_buses, bus_types):
    P_scheduled = np.zeros(num_buses)
    Q_scheduled = np.zeros(num_buses)
    V_scheduled = np.zeros(num_buses)
    delta_scheduled = np.zeros(num_buses)

    for i in range(num_buses):
        if bus_types[i] == 1:  # PQ bus
            while True:
                try:
                    P_scheduled[i] = float(input(f"Enter scheduled P for PQ bus {i+1}: "))
                    Q_scheduled[i] = float(input(f"Enter scheduled Q for PQ bus {i+1}: "))
                    break
                except ValueError:
                    print("Invalid input. Please enter a valid number.")

        elif bus_types[i] == 2:  # PV bus
            while True:
                try:
                    P_scheduled[i] = float(input(f"Enter scheduled P for PV bus {i+1}: "))
                    V_scheduled[i] = float(input(f"Enter specified V for PV bus {i+1}: "))
                    break
                except ValueError:
                    print("Invalid input. Please enter a valid number.")

        elif bus_types[i] == 3:  # Slack bus
            while True:
                try:
                    V_scheduled[i] = float(input(f"Enter specified V for slack bus {i+1}: "))
                    delta_scheduled[i] = float(input(f"Enter specified angle for slack bus {i+1} (in radians): "))
                    break
                except ValueError:
                    print("Invalid input. Please enter a valid number.")

    return P_scheduled, Q_scheduled, V_scheduled, delta_scheduled

def get_initial_guesses(num_buses, bus_types, V_scheduled, delta_scheduled):
    Vmag = np.zeros(num_buses)
    delta = np.zeros(num_buses)

    for i in range(num_buses):
        if bus_types[i] == 1:  # PQ bus
            while True:
                try:
                    Vmag[i] = float(input(f"Enter initial guess for voltage magnitude at PQ bus {i+1}: "))
                    delta[i] = float(input(f"Enter initial guess for voltage angle at PQ bus {i+1} (in radians): "))
                    break
                except ValueError:
                    print("Invalid input. Please enter a valid number.")

        elif bus_types[i] == 2:  # PV bus
            Vmag[i] = V_scheduled[i]
            while True:
                try:
                    delta[i] = float(input(f"Enter initial guess for voltage angle at PV bus {i+1} (in radians): "))
                    break
                except ValueError:
                    print("Invalid input. Please enter a valid number.")

        elif bus_types[i] == 3:  # Slack bus
            Vmag[i] = V_scheduled[i]
            delta[i] = delta_scheduled[i]

    return Vmag, delta
# Define the function to convert the impedance matrix to an admittance matrix
def convert_to_admittances(impedance_matrix):
    num_buses = impedance_matrix.shape[0]
    admittance_matrix = np.zeros((num_buses, num_buses), dtype=complex)

    for i in range(num_buses):
        for j in range(num_buses):
            if i != j and impedance_matrix[i, j] != 0:
                admittance_matrix[i, j] = -1 / impedance_matrix[i, j]

    for i in range(num_buses):
        admittance_matrix[i, i] = -np.sum(admittance_matrix[i, :])

    return admittance_matrix

def convert_admittance_to_polar(Ybus):
    num_buses = Ybus.shape[0]
    Ybus_mag = np.zeros((num_buses, num_buses))
    Ybus_angle = np.zeros((num_buses, num_buses))

    for i in range(num_buses):
        for j in range(num_buses):
            Ybus_mag[i, j] = np.abs(Ybus[i, j])
            Ybus_angle[i, j] = np.angle(Ybus[i, j])

    return Ybus_mag, Ybus_angle
def P_expression(bus, Ybus, V, delta):
    num_buses = len(Ybus)
    P = 0
    for k in range(num_buses):
        P += V[bus] * V[k] * sp.Abs(Ybus[bus, k]) * sp.cos(delta[k] - delta[bus] + sp.arg(Ybus[bus, k]))
    return P

def Q_expression(bus, Ybus, V, delta):
    num_buses = len(Ybus)
    Q = 0
    for k in range(num_buses):
        Q -= V[bus] * V[k] * sp.Abs(Ybus[bus, k]) * sp.sin(delta[k] - delta[bus] + sp.arg(Ybus[bus, k]))
    return Q

def symbolic_jacobian(Ybus, pq_indices, pv_indices):
    num_buses = Ybus.shape[0]
    V = sp.symbols(f'V:{num_buses}')
    delta = sp.symbols(f'delta:{num_buses}')

    # Initialize empty submatrices
    J1 = sp.zeros(len(pq_indices) + len(pv_indices), len(pq_indices) + len(pv_indices))
    J2 = sp.zeros(len(pq_indices) + len(pv_indices), len(pq_indices))
    J3 = sp.zeros(len(pq_indices), len(pq_indices) + len(pv_indices))
    J4 = sp.zeros(len(pq_indices), len(pq_indices))

    # Populate J1, J2, J3, J4 with symbolic derivatives
    for i, pi in enumerate(pq_indices + pv_indices):
        for j, pj in enumerate(pq_indices + pv_indices):
            J1[i, j] = sp.diff(P_expression(pi, Ybus, V, delta), delta[pj])

    for i, pi in enumerate(pq_indices + pv_indices):
        for j, pj in enumerate(pq_indices):
            J2[i, j] = sp.diff(P_expression(pi, Ybus, V, delta), V[pj])

    for i, pi in enumerate(pq_indices):
        for j, pj in enumerate(pq_indices + pv_indices):
            J3[i, j] = sp.diff(Q_expression(pi, Ybus, V, delta), delta[pj])

    for i, pi in enumerate(pq_indices):
        for j, pj in enumerate(pq_indices):
            J4[i, j] = sp.diff(Q_expression(pi, Ybus, V, delta), V[pj])

    # Combine J1, J2, J3, J4 into full Jacobian
    J = sp.Matrix.vstack(sp.Matrix.hstack(J1, J2), sp.Matrix.hstack(J3, J4))

    return J

def power_mismatch(Ybus, Vmag, delta, P_scheduled, Q_scheduled, bus_types):
    num_buses = len(Ybus)
    
    # Find indices of PQ and PV buses
    pq_indices = [i for i in range(num_buses) if bus_types[i] == 1]
    pv_indices = [i for i in range(num_buses) if bus_types[i] == 2]
    
    # Initialize mismatches
    delP = np.zeros(len(pq_indices) + len(pv_indices))
    delQ = np.zeros(len(pq_indices))
    
    # Compute mismatch for PQ buses
    for i, m in enumerate(pq_indices):
        P_calc = sum(Vmag[m] * Vmag[k] * np.abs(Ybus[m, k]) * np.cos(delta[k] - delta[m] + np.angle(Ybus[m, k])) for k in range(num_buses))
        Q_calc = -sum(Vmag[m] * Vmag[k] * np.abs(Ybus[m, k]) * np.sin(delta[k] - delta[m] + np.angle(Ybus[m, k])) for k in range(num_buses))
        delP[i] = P_scheduled[m] - P_calc
        delQ[i] = Q_scheduled[m] - Q_calc
    
    # Compute mismatch for PV buses (only real power P mismatch)
    for i, m in enumerate(pv_indices):
        P_calc = sum(Vmag[m] * Vmag[k] * np.abs(Ybus[m, k]) * np.cos(delta[k] - delta[m] + np.angle(Ybus[m, k])) for k in range(num_buses))
        delP[len(pq_indices) + i] = P_scheduled[m] - P_calc
    
    return delP, delQ

def update(Vmag, delta, delta_V, delta_delta, pq_indices, pv_indices):
    # Update angles for all buses
    delta[pq_indices + pv_indices] += delta_delta

    # Update voltage magnitudes for PQ buses only
    Vmag[pq_indices] += delta_V

    return Vmag, delta

def newton_raphson_sympy(Ybus, P_scheduled, Q_scheduled, V_scheduled, delta_scheduled, bus_types, tol=0.01, max_iter=100):
    num_buses = len(Ybus)
    
    pq_indices = [i for i in range(num_buses) if bus_types[i] == 1]
    pv_indices = [i for i in range(num_buses) if bus_types[i] == 2]
    
    Vmag, delta = get_initial_guesses(num_buses, bus_types, V_scheduled, delta_scheduled)
    
    for iteration in range(max_iter):
        # Calculate power mismatches
        delP, delQ = power_mismatch(Ybus, Vmag, delta, P_scheduled, Q_scheduled, bus_types)
        
        # Check for convergence before calculating the Jacobian
        norm_delP = np.linalg.norm(delP)
        norm_delQ = np.linalg.norm(delQ)
        print(f"Power Mismatch (delP): {delP}")
        print(f"Power Mismatch (delQ): {delQ}")
        print(f"Norm of delP: {norm_delP}")
        print(f"Norm of delQ: {norm_delQ}")
        
        if norm_delP < tol and norm_delQ < tol:
            print(f"Converged in {iteration+1} iterations.")
            break
        
        # Calculate the Jacobian matrix based on current Vmag and delta
        J_symbolic = symbolic_jacobian(Ybus, pq_indices, pv_indices)

        # Create a dictionary for all substitutions
        subs_dict = {f'V{i}': Vmag[i] for i in range(num_buses)}
        subs_dict.update({f'delta{i}': delta[i] for i in range(num_buses)})
        
        # Substitute numerical values into the symbolic Jacobian
        J_numeric = np.array(J_symbolic.subs(subs_dict)).astype(np.float64)

        # Print Jacobian matrix
        print(f"Jacobian Matrix (Iteration {iteration + 1}):")
        print(J_numeric)

        # Solve for voltage and angle updates
        delta_Vdelta = np.linalg.solve(J_numeric, np.concatenate([delP, delQ]))
        delta_delta = delta_Vdelta[:len(pq_indices) + len(pv_indices)]
        delta_V = delta_Vdelta[len(pq_indices) + len(pv_indices):]

        # Update angles for all buses
        delta[pq_indices + pv_indices] += delta_delta

        # Update voltage magnitudes for PQ buses only
        Vmag[pq_indices] += delta_V
        
        # Recalculate power mismatches with updated Vmag and delta
        delP, delQ = power_mismatch(Ybus, Vmag, delta, P_scheduled, Q_scheduled, bus_types)
        print(f"Recalculated Power Mismatch (delP): {delP}")
        print(f"Recalculated Power Mismatch (delQ): {delQ}")
    
    return Vmag, delta

def main():
    while True:
        try:
            num_buses = int(input("Enter the number of buses: "))
            if num_buses <= 0:
                raise ValueError
            break
        except ValueError:
            print("Invalid input. Please enter a positive integer.")

    impedance_matrix, bus_types = get_impedances_and_bus_types(num_buses)
    Ybus = convert_to_admittances(impedance_matrix)
    Ybus_mag, Ybus_angle = convert_admittance_to_polar(Ybus)

    P_scheduled, Q_scheduled, V_scheduled, delta_scheduled = get_inputs(num_buses, bus_types)
    
    Vmag, delta = newton_raphson_sympy(Ybus, P_scheduled, Q_scheduled, V_scheduled, delta_scheduled, bus_types)

    print("Admittance Matrix (Ybus):")
    print(Ybus)
    print("Admittance Matrix Magnitude:")
    print(Ybus_mag)
    print("Admittance Matrix Angle (radians):")
    print(Ybus_angle)
    print("Final Voltage Magnitudes:")
    print(Vmag)
    print("Final Voltage Angles (radians):")
    print(delta)

if __name__ == "__main__":
    main()
