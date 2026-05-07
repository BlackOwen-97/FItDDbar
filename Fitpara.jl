using IMinuit

function make_goal(lam; every::Int=10, logio::Union{Nothing,IO}=nothing)
    neval = Ref(0)

    return function (para)
        neval[] += 1
        value = goalminnicedipole(para, lam)

        if neval[] % every == 0
            msg = "neval=$(neval[])  para=$(para)  chisq=$(value)"
            println(msg)
            if logio !== nothing
                println(logio, msg)
                flush(logio)
            end
        end

        return value
    end
end

function fitpara(para0, lam; logfile::String="fit_trace.txt", every::Int=10)
    NP = length(para0)
    para  = zeros(NP)
    error = zeros(NP)

    step = 0.1 .* abs.(para0)
    step[step .== 0.0] .= 0.1

    open(logfile, "w") do io
        println(io, "# Minuit trace log")
        println(io, "# print/save every = $every function evaluations")
        flush(io)

        goal = make_goal(lam; every=every, logio=io)

        fit = Minuit(goal, para0; error=step)
        fit.strategy = 1

        migrad(fit)

        para  .= fit.values
        error .= fit.errors
        chisq  = fit.fval

        println(io, "FINAL  para=$(para)  err=$(error)  chisq=$(chisq)")
        flush(io)

        return para, error, chisq
    end
end