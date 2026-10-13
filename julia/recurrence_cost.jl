# Стоимость одной оценки начального условия рекуррентным методом Attractors.jl (DynamicalSystems.jl)
# на общих начальных условиях results/e59/ics_<задача>.csv (их пишет experiments/e59_eval_cost.py export).
#
# Для каждой задачи и каждой настройки интегратора (адаптивный Tsit5 и наш РК4 с тем же шагом dt):
#   * сопоставитель AttractorsViaRecurrences на сетке, покрывающей область и аттракторы (разреженная сетка);
#   * начальные условия обрабатываются по одному, как в перепечатанной перебором выборке; для каждого — время,
#     метка; найденные аттракторы описываются статистиками точек; цель узнаётся предикатом по этим точкам;
#   * доли меток сравниваются с нашей переписью (проверка, что настройка не порождает ложных аттракторов).
# Запуск: julia julia/recurrence_cost.jl [n_ic] [задачи...]   → results/e59/cost_julia.json
using DynamicalSystems, OrdinaryDiffEq, JSON3, DelimitedFiles, Statistics
const HAVE_RK4 = try
    @eval using OrdinaryDiffEqLowOrderRK
    true
catch
    false
end

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUT = joinpath(ROOT, "results", "e59")

function chua_rule(u, p, t)
    a, b, g, m0, m1 = p
    x, y, z = u
    fx = m1 * x + 0.5 * (m0 - m1) * (abs(x + 1) - abs(x - 1))
    return SVector(a * (y - x - fx), x - y + z, -(b * y + g * z))
end

function coupled_chua_rule(u, p, t)
    a, b, g, m0, m1, eps = p
    x1, y1, z1, x2, y2, z2 = u
    f1 = m1 * x1 + 0.5 * (m0 - m1) * (abs(x1 + 1) - abs(x1 - 1))
    f2 = m1 * x2 + 0.5 * (m0 - m1) * (abs(x2 + 1) - abs(x2 - 1))
    m = 0.5 * (x1 + x2)
    return SVector(a * (y1 - x1 - f1) + eps * (m - x1), x1 - y1 + z1, -(b * y1 + g * z1),
                   a * (y2 - x2 - f2) + eps * (m - x2), x2 - y2 + z2, -(b * y2 + g * z2))
end

se11_rule(u, p, t) = SVector(u[2], -u[1] + u[2] * u[3], -u[2] - 12.0 * u[3] + u[1]^2 + 9.0 * u[1] * u[3] - 1.0)
le1_rule(u, p, t) = SVector(u[2], -u[1] + u[2] * u[3], -u[1] - p[1] * u[1] * u[2] - p[2] * u[1] * u[3])

# цель по точкам найденного аттрактора (матрица n × d; хранится по одной точке на ячейку сетки, поэтому
# аттрактор может быть покрыт частично — критерии по размаху мягкие, равновесия отсекаются размахом)
function chua_target(M)
    size(M, 1) >= 3 && maximum(abs.(M[:, 1])) < 3.0 && (maximum(M[:, 1]) - minimum(M[:, 1])) > 0.3
end
function coupled_target(M)
    size(M, 1) < 3 && return false
    ha(i) = maximum(abs.(M[:, i])) < 3.0 && (maximum(M[:, i]) - minimum(M[:, i])) > 0.3
    still(i) = (maximum(M[:, i]) - minimum(M[:, i])) < 0.1
    (ha(1) && still(4)) || (ha(4) && still(1)) || (ha(1) && ha(4))
end
bounded_osc_target(M) = size(M, 1) >= 3 && (maximum(M[:, 1]) - minimum(M[:, 1])) > 0.3

const PROBLEMS = Dict(
    # lost — шагов Δt подряд вне сетки до метки «ушла»: у ограниченных систем (Чуа) сетка шире области и переходные
    # процессы могут ненадолго выходить за неё, поэтому порог большой; у систем с уходом на бесконечность — малый
    "chua_II" => (rule = chua_rule, p = [8.4, 12.0, -0.005, -1.2, -0.05], dt = 0.0025,
                  grid = ([-20.0, -12.0, -60.0], [20.0, 12.0, 60.0]), target = chua_target, lost = 2000),
    "coupled_chua_II_eps0.1" => (rule = coupled_chua_rule, p = [8.4, 12.0, -0.005, -1.2, -0.05, 0.1], dt = 0.0025,
                  grid = ([-20.0, -12.0, -60.0, -20.0, -12.0, -60.0], [20.0, 12.0, 60.0, 20.0, 12.0, 60.0]),
                  target = coupled_target, lost = 2000),
    "se11" => (rule = se11_rule, p = [0.0], dt = 0.01, grid = ([-12.0, -12.0, -12.0], [12.0, 12.0, 12.0]),
               target = bounded_osc_target, lost = 100),
    "le1" => (rule = le1_rule, p = [15.0, 1.0], dt = 0.01, grid = ([-12.0, -12.0, -12.0], [12.0, 12.0, 12.0]),
              target = bounded_osc_target, lost = 100),
)

function make_mapper(ds, lo, hi; ncell, Δt, rec, locate, lost)
    grid = ntuple(i -> range(lo[i], hi[i]; length = ncell + 1), length(lo))
    return AttractorsViaRecurrences(ds, grid; Δt = Δt, sparse = true,
        consecutive_recurrences = rec, attractor_locate_steps = locate,
        consecutive_lost_steps = lost, horizon_limit = 1e3, maximum_iterations = 20_000_000)
end

function describe(A)
    M = Matrix(A)
    return Dict("n" => size(M, 1), "mean" => vec(mean(M, dims = 1)), "min" => vec(minimum(M, dims = 1)),
                "max" => vec(maximum(M, dims = 1)))
end

function run_problem(name, n_ic, settings, out, out_path)
    P = PROBLEMS[name]
    ics = readdlm(joinpath(OUT, "ics_$(name).csv"), ',')
    n_ic = min(n_ic, size(ics, 1))
    lo, hi = P.grid
    d = length(lo)
    res = Dict{String,Any}()
    for st in settings
        fixed_alg = HAVE_RK4 ? RK4() : Tsit5()
        diffeq = st.adaptive ? (alg = Tsit5(), abstol = 1e-9, reltol = 1e-9) :
                               (alg = fixed_alg, adaptive = false, dt = P.dt)
        ds = CoupledODEs(P.rule, SVector{d}(ics[1, :]), P.p; diffeq)
        mapper = make_mapper(ds, lo, hi; ncell = st.ncell, Δt = st.Δt, rec = st.rec, locate = st.locate, lost = P.lost)
        mapper(SVector{d}(ics[1, :]))                      # компиляция
        ds2 = CoupledODEs(P.rule, SVector{d}(ics[1, :]), P.p; diffeq)
        mapper = make_mapper(ds2, lo, hi; ncell = st.ncell, Δt = st.Δt, rec = st.rec, locate = st.locate, lost = P.lost)
        times = Float64[]; labels = Int[]
        target_labels = Set{Int}(); first_hit = 0; t_first_hit = 0.0
        t_total = 0.0
        for i in 1:n_ic
            u0 = SVector{d}(ics[i, :])
            t = @elapsed lab = mapper(u0)
            t_total += t
            push!(times, t); push!(labels, lab)
            if lab > 0 && !(lab in target_labels)
                atts = extract_attractors(mapper)
                if haskey(atts, lab) && P.target(Matrix(atts[lab]))
                    push!(target_labels, lab)
                end
            end
            if first_hit == 0 && lab in target_labels
                first_hit = i; t_first_hit = t_total
            end
        end
        # пробные точки: состояния на целевых аттракторах (проверка, что сопоставитель и предикат их узнают)
        probe_path = joinpath(OUT, "probe_$(name).csv")
        probe = Dict{String,Any}()
        if isfile(probe_path)
            PR = readdlm(probe_path, ',')
            PR = ndims(PR) == 1 ? reshape(PR, 1, :) : PR
            for i in 1:size(PR, 1)
                tp = @elapsed lab = mapper(SVector{d}(PR[i, :]))
                atts_now = extract_attractors(mapper)
                ok = lab > 0 && haskey(atts_now, lab) && P.target(Matrix(atts_now[lab]))
                probe["probe$(i)"] = Dict("label" => lab, "is_target" => ok, "ms" => 1e3 * tp)
            end
        end
        atts = extract_attractors(mapper)
        summ = Dict(string(k) => describe(A) for (k, A) in atts)
        fr = Dict{String,Float64}()
        for l in unique(labels)
            fr[string(l)] = count(==(l), labels) / n_ic
        end
        key = (st.adaptive ? "tsit5" : "rk4") * "_c$(st.ncell)_dt$(st.Δt)_r$(st.rec)"
        res[key] = Dict("adaptive" => st.adaptive, "alg" => string(typeof(diffeq.alg)), "ncell" => st.ncell,
                        "Dt" => st.Δt, "rec" => st.rec,
                        "n_ic" => n_ic, "ms_per_ic" => 1e3 * t_total / n_ic, "ms_median" => 1e3 * median(times),
                        "ms_max" => 1e3 * maximum(times), "fractions" => fr, "attractors" => summ,
                        "target_labels" => collect(target_labels), "first_hit_ic" => first_hit,
                        "t_first_hit_s" => t_first_hit, "n_attractors" => length(atts), "probe" => probe)
        out[name] = res
        open(out_path, "w") do io
            JSON3.pretty(io, out)
        end
        println(name, " ", key, ": ", round(1e3 * t_total / n_ic; digits = 3), " ms/IC, attractors ", length(atts),
                ", fractions ", fr, ", target ", collect(target_labels), " first hit ", first_hit, " probe ", probe)
        flush(stdout)
    end
    return res
end

function main()
    n_ic = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 500
    names = length(ARGS) >= 2 ? ARGS[2:end] : collect(keys(PROBLEMS))
    settings = [(adaptive = true, ncell = 100, Δt = 0.1, rec = 100, locate = 1000),
                (adaptive = true, ncell = 100, Δt = 0.1, rec = 1000, locate = 1000),
                (adaptive = false, ncell = 100, Δt = 0.1, rec = 1000, locate = 1000)]
    out_path = joinpath(OUT, "cost_julia.json")
    out = isfile(out_path) ? copy(JSON3.read(read(out_path, String), Dict{String,Any})) : Dict{String,Any}()
    for name in names
        run_problem(name, n_ic, settings, out, out_path)
    end
    println("saved ", out_path)
end

main()
