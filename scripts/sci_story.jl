#!/usr/bin/env julia
using DifferentialEquations
using JSON

function latent_dynamics!(du, u, p, t)
    alpha, beta, gamma, target = p
    gating = gamma * sin(beta * t)
    du[1] = -alpha * (u[1] - target) + gating + 0.03 * cos(0.8 * t)
end

function story_from_solution(sol)
    values = [u[1] for u in sol.u]
    delta = values[end] - values[1]
    slope = round(delta / sol.t[end], digits=3)
    confidence = clamp(0.9 - abs(slope), 0.3, 0.95)
    status = slope < 0 ? "Cooling" : "Heating"
    note = slope < 0 ? "The latent is settling; gating keeps magnitude in check." : "Latent energy is rising; decoder drift may grow."
    return Dict(
        "trajectory" => sol.t,
        "values" => values,
        "story" => Dict(
            "status" => status,
            "slope" => slope,
            "confidence" => confidence,
            "note" => note
        )
    )
end

function run_simulation(; alpha=0.12, beta=0.35, gamma=0.07, target=0.0, tspan=(0.0, 12.0))
    u0 = [1.0]
    params = (alpha, beta, gamma, target)
    prob = ODEProblem(latent_dynamics!, u0, tspan, params)
    sol = solve(prob, Tsit5(), reltol=1e-6, abstol=1e-6, saveat=0.1)
    return sol
end

function main()
    sol = run_simulation()
    payload = story_from_solution(sol)
    mkpath("artifacts")
    open("artifacts/sci_story.json", "w") do io
        JSON.print(io, payload)
    end
    println("Saved simulation story to artifacts/sci_story.json")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
