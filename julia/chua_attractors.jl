# Независимая базовая линия: Attractors.jl (DynamicalSystems.jl) на цепи Чуа, набор I.
# Доли бассейнов при выборке в шарах радиуса 3 и 5 (ср. табл. 1 Guan & Xie 2025) и в боксе Зелинки.
using DynamicalSystems, OrdinaryDiffEq, JSON3, Random

function chua_rule(u, p, t)
    a, b, g, m0, m1 = p
    x, y, z = u
    fx = m1 * x + 0.5 * (m0 - m1) * (abs(x + 1) - abs(x - 1))
    return SVector(a * (y - x - fx), x - y + z, -(b * y + g * z))
end

p = [8.4562218418, 12.0732335925, 0.0051631393, -0.1767573476, -1.1467573476]
u0 = SVector(0.1698, 1.2067, 3.9548)
diffeq = (alg = Vern9(), abstol = 1e-10, reltol = 1e-10)
ds = CoupledODEs(chua_rule, u0, p; diffeq)

xg = range(-15.0, 15.0; length = 301)
yg = range(-6.0, 6.0; length = 301)
zg = range(-20.0, 20.0; length = 301)
mapper = AttractorsViaRecurrences(ds, (xg, yg, zg);
    Δt = 0.01, sparse = true,
    consecutive_recurrences = 2000, attractor_locate_steps = 2000,
    consecutive_lost_steps = 2000, horizon_limit = 1e3, maximum_iterations = 2_000_000)

results = Dict{String,Any}()
function summarize(mapper, fs)
    atts = extract_attractors(mapper)
    summ = Dict{String,Any}()
    for (k, A) in atts
        M = Matrix(A)
        summ[string(k)] = Dict("n" => size(M, 1), "mean" => vec(sum(M, dims = 1) ./ size(M, 1)),
            "min" => vec(minimum(M, dims = 1)), "max" => vec(maximum(M, dims = 1)))
    end
    return Dict{String,Any}("fractions" => Dict(string(k) => v for (k, v) in fs), "attractors" => summ)
end

for (name, region, N) in [("r3", HSphere(3.0, 3), 10000), ("r5", HSphere(5.0, 3), 10000),
                           ("box_zelinka", HRectangle([-10.0, -10.0, -15.0], [10.0, 10.0, 15.0]), 20000)]
    sampler = RandomICsSampler(N, region, 1234)
    t = @elapsed fs = basins_fractions(mapper, sampler; show_progress = false)
    r = summarize(mapper, fs)
    r["time_s"] = t
    results[name] = r
    println(name, ": ", fs, "  (", round(t; digits = 1), " s)")
    # сброс сопоставителя, чтобы метки не накапливались между областями
    global mapper = AttractorsViaRecurrences(ds, (xg, yg, zg);
        Δt = 0.01, sparse = true,
        consecutive_recurrences = 2000, attractor_locate_steps = 2000,
        consecutive_lost_steps = 2000, horizon_limit = 1e3, maximum_iterations = 2_000_000)
end

open(joinpath(@__DIR__, "..", "results", "julia_chua_attractors.json"), "w") do io
    JSON3.pretty(io, results)
end
println("saved results/julia_chua_attractors.json")
