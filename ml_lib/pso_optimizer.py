import numpy as np

def pso_optimize(model_func, X_train, y_train, param_grid, update_progress):
    """
    Particle Swarm Optimization (PSO) untuk optimasi hyperparameter sederhana.
    model_func: fungsi yang menerima dict param → return akurasi (float)
    param_grid: dict batas parameter, contoh:
        {
            'learning_rate': (0.01, 0.3),
            'max_depth': (3, 10)
        }
    """

    update_progress(10, "⚙️ Inisialisasi swarm PSO...")

    num_particles = 8
    max_iter = 10
    w, c1, c2 = 0.7, 1.5, 1.5  # inertia, cognitive, social factors

    params = list(param_grid.keys())
    lb = np.array([v[0] for v in param_grid.values()])
    ub = np.array([v[1] for v in param_grid.values()])

    # Inisialisasi posisi dan kecepatan
    pos = np.random.uniform(lb, ub, (num_particles, len(params)))
    vel = np.zeros_like(pos)
    pbest = pos.copy()
    pbest_scores = np.zeros(num_particles)

    # Evaluasi awal
    for i in range(num_particles):
        score = model_func(dict(zip(params, pos[i])))
        pbest_scores[i] = score

    gbest = pbest[np.argmax(pbest_scores)].copy()
    gbest_score = np.max(pbest_scores)

    for it in range(max_iter):
        for i in range(num_particles):
            r1, r2 = np.random.rand(2)
            vel[i] = (
                w * vel[i]
                + c1 * r1 * (pbest[i] - pos[i])
                + c2 * r2 * (gbest - pos[i])
            )
            pos[i] += vel[i]
            pos[i] = np.clip(pos[i], lb, ub)

            score = model_func(dict(zip(params, pos[i])))
            if score > pbest_scores[i]:
                pbest[i] = pos[i].copy()
                pbest_scores[i] = score

        # Update global best
        best_idx = np.argmax(pbest_scores)
        if pbest_scores[best_idx] > gbest_score:
            gbest = pbest[best_idx].copy()
            gbest_score = pbest_scores[best_idx]

        update_progress(
            20 + int((it + 1) / max_iter * 70),
            f"🌀 Iterasi PSO {it+1}/{max_iter} - Akurasi terbaik: {gbest_score:.4f}",
        )

    update_progress(100, f"✅ PSO selesai - Akurasi optimal: {gbest_score:.4f}")

    best_params_
