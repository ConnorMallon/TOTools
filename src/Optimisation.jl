# #=
function execute_optimisation(optim_function,Vol_constr_ub,Vol_constr_lb,p00,iterations,optimiser::OptimiserType{:MMA},problem)
  opt = NLoptAlg(:LD_MMA)
  options = NLoptOptions(xtol_rel=-1,ftol_rel=-1,xtol_abs=-1,ftol_abs=-1,maxeval=iterations)
  execute_with_nonconvex(optim_function,Vol_constr_ub,Vol_constr_lb,p00,iterations,opt,options,problem)
end

function execute_optimisation(optim_function,Vol_constr_ub,Vol_constr_lb,p00,iterations,optimiser::OptimiserType{:IPOPT},problem)
  opt = IpoptAlg()
  options = IpoptOptions( max_iter = iterations )
  execute_with_nonconvex(optim_function,Vol_constr_ub,Vol_constr_lb,p00,iterations,opt,options,problem)
end

function execute_with_nonconvex(optim_function,Vol_constr_ub,Vol_constr_lb,p00,iterations,opt,options,problem::ProblemType{:heat_simp})
  lb = 0
  ub = 1
  execute_with_nonconvex(optim_function,Vol_constr_ub,Vol_constr_lb,p00,iterations,opt,options,problem,lb,ub)
end

execute_with_nonconvex(optim_function,Vol_constr_ub,Vol_constr_lb,p00,iterations,opt,options,problem::ProblemType{:cantilever_simp}) = execute_with_nonconvex(optim_function,Vol_constr_ub,Vol_constr_lb,p00,iterations,opt,options,ProblemType{Symbol("heat_simp")}())
execute_with_nonconvex(optim_function,Vol_constr_ub,Vol_constr_lb,p00,iterations,opt,options,problem::ProblemType{:MBB_simp}) = execute_with_nonconvex(optim_function,Vol_constr_ub,Vol_constr_lb,p00,iterations,opt,options,ProblemType{Symbol("heat_simp")}())

function execute_with_nonconvex(optim_function,Vol_constr_ub,Vol_constr_lb,p00,iterations,opt,options,problem)
  lb = -1e4
  ub = 1e4
  execute_with_nonconvex(optim_function,Vol_constr_ub,Vol_constr_lb,p00,iterations,opt,options,problem,lb,ub)
end

function execute_with_nonconvex(optim_function,Vol_constr_ub,Vol_constr_lb,p00,iterations,opt,options,problem,lb,ub)
  js=[]
  is=[]
  ts=[]
  i=0
  start_time = time()
  function logging(j)
    push!(js,j)
    @show j
    i += 1
    push!(is,i)
    @show i
    t = time()-start_time
    push!(ts,t)
  end
  function optim_function_with_logging(p)
    j = optim_function(p)
    Zygote.ignore() do 
        logging(j)
    end
    j
  end
  convcriteria = Nonconvex.KKTCriteria()
  convcriteria = Nonconvex.ScaledKKTCriteria()
  T = eltype(p00)
  nvars = length(p00)
  x0 = copy(p00)
  model_opt = Nonconvex.Model(optim_function_with_logging)
  addvar!(model_opt, lb*ones(T, nvars), ub*ones(T, nvars)) #
  add_ineq_constraint!(model_opt, Nonconvex.FunctionWrapper(Vol_constr_ub, length(Vol_constr_ub(x0))))
  add_ineq_constraint!(model_opt, Nonconvex.FunctionWrapper(Vol_constr_lb, length(Vol_constr_lb(x0))))
  res = Nonconvex.optimize(model_opt, opt, x0; options = options,  plot_trace = true)
  fcalls = res.fcalls
  gcalls = 9999#res.gcalls
  iters = length(is)
  jf = res.minimum
  pf = res.minimizer
  fcalls,gcalls,iters,jf,pf,js,ts
end

function execute_optimisation(optim_function,Vol_constr_ub,Vol_constr_lb,p0,iterations,optimiser::OptimiserType{:ADAM},problem)
  js=[]
  is=[]
  ts=[]
  i=0
  start_time = time()
  function logging(j)
    push!(js,j)
    @show j
    i += 1
    push!(is,i)
    @show i
    t = time()-start_time
    push!(ts,t)
  end
  function g(p)
    j,∇ = Zygote.withgradient(optim_function,p)
    logging(j)
    ∇
  end
  n_ADAM_epochs = iterations
  opt=ADAM()
  for i in 1:n_ADAM_epochs
    Flux.update!(opt,p0,collect(Iterators.flatten(g(p0))))
  end
  jf = optim_function(p0)
  (fcalls,g_calls,iters,jf,p,js,ts) = (n_ADAM_epochs,n_ADAM_epochs,n_ADAM_epochs,jf,p0,js,ts)
end

function execute_optimisation(optim_function,Vol_constr_ub,Vol_constr_lb,p0,iterations,optimiser::OptimiserType{:LBFGS},problem)
  js=[]
  is=[]
  ts=[]
  i=0
  start_time = time()
  function logging(j)
    push!(js,j)
    @show j
    i += 1
    push!(is,i)
    t = time()-start_time
    push!(ts,t)
  end
  function fg!(F,G,p)
    jp,jpb = Zygote.pullback(optim_function,p)
    logging(jp)

    if G != nothing
      dj=1
      grads_vec = collect(Iterators.flatten(jpb(dj)[1]))
      copyto!(G,grads_vec)
    end
    loss_vec = collect(Iterators.flatten(jp))[1]
    return loss_vec
  end
  optimizer = LBFGS(linesearch=LineSearches.BackTracking())#HagerZhang())#,alphaguess=InitialQuadratic())
  @show res = Optim.optimize(
                              Optim.only_fg!(fg!), p0, 
                              optimizer,
                              Optim.Options(
                                  iterations = iterations, store_trace=true, show_trace=true,
                                  f_abstol = -1.0, f_reltol = -1.0, #-1.0, f_reltol = -1.0, 
                                  x_reltol = -1e-12, x_abstol = -1e-12,
                                  allow_f_increases=true,
                                  )
                              )
  js = Optim.f_trace(res)
  fcalls = Optim.f_calls(res)
  gcalls = Optim.g_calls(res)
  iters = Optim.iterations(res)
  jf = Optim.minimum(res)
  p = Optim.minimizer(res)
  (fcalls,gcalls,iters,jf,p,js,ts)
end
# =#










#=

function execute_optimisation(optim_function,Vol_constr_ub,Vol_constr_lb,p00,iterations,optimiser::Union{OptimiserType{:MMA},OptimiserType{:IPOPT}},problem::ProblemType{:heat_simp})
  lb = 0#1e-9#-1e4#qlb
  ub = 1#-1e-9#1e4#qub
  execute_optimisation(optim_function,Vol_constr_ub,Vol_constr_lb,p00,iterations,optimiser,problem,lb,ub)
end

function execute_optimisation(optim_function,Vol_constr_ub,Vol_constr_lb,p00,iterations,optimiser::OptimiserType{:MMA},TOmethod,lb,ub)



opt = NLoptAlg(:LD_MMA)
options = NLoptOptions(xtol_rel=-1,ftol_rel=-1,xtol_abs=-1,ftol_abs=-1,maxeval=iterations)

@show opt


 js=[]
 is=[]
 ts=[]
 i=0
 start_time = time()
 function logging(j)
     push!(js,j)
     @show j
     #@show js
     i += 1
     push!(is,i)
     @show i
     t = time()-start_time
     push!(ts,t)
 end

 function optim_function_with_logging(p)
     j = optim_function(p)
     Zygote.ignore() do 
         logging(j)
     end
     j
 end

 @show Vol_constr_ub(p00)


 convcriteria = Nonconvex.KKTCriteria()
 convcriteria = Nonconvex.ScaledKKTCriteria()#GenericCriteria()
 
 T = eltype(p00)
 nvars = length(p00)
 x0 = copy(p00)

 #=
 model_opt = Nonconvex.Model(optim_function_with_logging)
 addvar!(model_opt, lb*ones(T, nvars), ub*ones(T, nvars)) #
 #add_eq_constraint!(model_opt, Nonconvex.FunctionWrapper(Vol_constr_ub, length(Vol_constr_ub(x0))))
 add_ineq_constraint!(model_opt, Nonconvex.FunctionWrapper(Vol_constr_ub, length(Vol_constr_ub(x0))))
 add_ineq_constraint!(model_opt, Nonconvex.FunctionWrapper(Vol_constr_lb, length(Vol_constr_lb(x0))))
 res = Nonconvex.optimize(model_opt, opt, x0; options = options,  plot_trace = true)

 fcalls = res.fcalls
 gcalls = 9999#res.gcalls
 iters = length(is)
 jf = res.minimum
 pf = res.minimizer

 fcalls,gcalls,iters,jf,pf,js,ts

 =#

 # #=

 function optim_function_with_logging_nlopt(p::Vector, grad::Vector)
     jp,jpb = Zygote.pullback(optim_function_with_logging,p)
     if length(grad) > 0
         grads_vec = collect(Iterators.flatten(jpb(1)[1]))
         copyto!(grad,grads_vec)
     end
     return jp
 end

 function Vol_constr_ub_nlopt(p::Vector,grad::Vector)
     if length(grad) > 0
         j,grad_vec = Zygote.withgradient(Vol_constr_ub,p)
         copyto!(grad,collect(Iterators.flatten(grad_vec)))
     end
     return j
 end

 function Vol_constr_lb_nlopt(p::Vector,grad::Vector)
     if length(grad) > 0
         j,grad_vec = Zygote.withgradient(Vol_constr_lb,p)
         copyto!(grad,collect(Iterators.flatten(grad_vec)))
     end
     return j
 end

 #grads_vec = collect(Iterators.flatten(jpb(dj)[1]))
 #        copyto!(G,grads_vec)

 #using NLopt
 
 opt = Opt(:LD_MMA,length(x0))
 opt.lower_bounds = lb*ones(T, nvars)
 opt.upper_bounds=ub*ones(T, nvars)
 opt.params["inner_maxeval"] = 10
 opt.params["verbosity"] =1
 #opt.maxeval=500
 opt.xtol_rel=-1
 opt.ftol_rel=-1
 opt.xtol_abs=-1
 opt.ftol_abs=-1
 
 opt.min_objective = optim_function_with_logging_nlopt
 inequality_constraint!(opt, Vol_constr_ub_nlopt, length(Vol_constr_ub(x0)) )
 inequality_constraint!(opt, Vol_constr_ub_nlopt, length(Vol_constr_lb(x0)))
 
 println("startn opt")
 (minf,minx,ret) = NLopt.optimize(opt, x0)
 numevals = opt.numevals # the number of function evaluations
 @show ret
 #println("got $minf at $minx after $numevals iterations (returned $ret)")

 # =#
 1

 numevals,numevals,numevals,minf,minx,js,ts


#execute_with_nonconvex(optim_function,Vol_constr_ub,Vol_constr_lb,p00,iterations,opt,options,TOmethod)
end

function execute_optimisation(optim_function,Vol_constr_ub,Vol_constr_lb,p00,iterations,optimiser::OptimiserType{:IPOPT},TOmethod,lb,ub)
opt = IpoptAlg()
options = IpoptOptions( max_iter = iterations )

js=[]
is=[]
ts=[]
i=0
start_time = time()
function logging(j)
    push!(js,j)
    @show j
    #@show js
    i += 1
    push!(is,i)
    @show i
    t = time()-start_time
    push!(ts,t)
end

function optim_function_with_logging(p)
    j = optim_function(p)
    Zygote.ignore() do 
        logging(j)
    end
    j
end

@show Vol_constr_ub(p00)

convcriteria = Nonconvex.KKTCriteria()
convcriteria = Nonconvex.ScaledKKTCriteria()#GenericCriteria()

T = eltype(p00)
nvars = length(p00)
x0 = copy(p00)

model_opt = Nonconvex.Model(optim_function_with_logging)
addvar!(model_opt, lb*ones(T, nvars), ub*ones(T, nvars)) #
#add_eq_constraint!(model_opt, Nonconvex.FunctionWrapper(Vol_constr_ub, length(Vol_constr_ub(x0))))
add_ineq_constraint!(model_opt, Nonconvex.FunctionWrapper(Vol_constr_ub, length(Vol_constr_ub(x0))))
add_ineq_constraint!(model_opt, Nonconvex.FunctionWrapper(Vol_constr_lb, length(Vol_constr_lb(x0))))
res = Nonconvex.optimize(model_opt, opt, x0; options = options,  plot_trace = true)

fcalls = res.fcalls
gcalls = 9999#res.gcalls
iters = length(is)
jf = res.minimum
pf = res.minimizer

fcalls,gcalls,iters,jf,pf,js,ts
#execute_with_nonconvex(optim_function,Vol_constr_ub,Vol_constr_lb,p00,iterations,opt,options,TOmethod)
end


=#
