module SingleIterationModule

using ADTypes: AutoEnzyme
using DynamicExpressions: AbstractExpression, string_tree, simplify_tree!, combine_operators
using ..UtilsModule: @threads_if
using ..CoreModule: AbstractOptions, Dataset, RecordType, create_expression, batch
using ..ComplexityModule: compute_complexity
using ..PopMemberModule: generate_reference
using ..PopulationModule: Population, finalize_costs
using ..PopMemberModule: PopMember
using ..HallOfFameModule: HallOfFame
using ..AdaptiveParsimonyModule: RunningSearchStatistics
using ..RegularizedEvolutionModule: reg_evol_cycle
using ..LossFunctionsModule: eval_cost
using ..ConstantOptimizationModule: optimize_constants
using ..RecorderModule: @recorder

# Cycle through regularized evolution many times,
# printing the fittest equation every 10% through
function s_r_cycle(
    dataset::D,
    pop::P,
    ncycles::Int,
    curmaxsize::Int,
    running_search_statistics::RunningSearchStatistics;
    verbosity::Int=0,
    options::AbstractOptions,
    record::RecordType,
    iteration
)::Tuple{
    P,HallOfFame{T,L,N},Float64
} where {T,L,D<:Dataset{T,L},N<:AbstractExpression{T},P<:Population{T,L,N}}
    max_temp = 1.0
    min_temp = 0.0
    if !options.annealing
        min_temp = max_temp
    end
    all_temperatures = ncycles > 1 ? LinRange(max_temp, min_temp, ncycles) : [max_temp]
    best_examples_seen = HallOfFame(options, dataset)
    num_evals = 0.0

    batched_dataset = options.batching ? batch(dataset, options.batch_size) : dataset
    new_pop = PopMember[]

    mutation_start_time = time()
    num_mutations = 0

    for temperature in all_temperatures
        new_pop_iter, tmp_num_evals = reg_evol_cycle(
            batched_dataset,
            pop,
            temperature,
            curmaxsize,
            running_search_statistics,
            options,
            record,
        )
        num_evals += tmp_num_evals
        append!(new_pop, new_pop_iter)
        num_mutations += length(new_pop_iter)
    end

    mutation_time = time() - mutation_start_time
    mutation_time_avg = mutation_time / num_mutations

    old_pop = PopMember[]
    replacement_ratio = hasproperty(options, :replacement_ratio) ? options.replacement_ratio : 0.5
    
    sort!(pop.members, by=p->p.loss)
    prev_loss = -1
    for member in pop.members
        if member.loss == prev_loss
            continue
        end
        push!(old_pop, member)
        prev_loss = member.loss
    end
    k = length(pop.members)
    for member in old_pop
        pop.members[k] = member
        k -= 1
        if k <= pop.n * replacement_ratio
            break
        end
    end
    
    sort!(new_pop, by=p->p.loss)
    k = 1
    prev_loss = -1
    for member in new_pop
        if member.loss == prev_loss
            continue
        end
        pop.members[k] = member
        k += 1
        prev_loss = member.loss
        if k > pop.n * replacement_ratio
            break
        end
    end

    println("pop member after mutation: ")
    for member in pop.members
        println(member.loss)
    end

    for member in pop.members
        size = compute_complexity(member, options)
        if 0 < size <= options.maxsize && (
            !best_examples_seen.exists[size] ||
            member.cost < best_examples_seen.members[size].cost
        )
            best_examples_seen.exists[size] = true
            best_examples_seen.members[size] = copy(member)
        end
    end

    @recorder begin
        record["complexity_stats"]["iteration$(iteration)"]["num_mutations"] = num_mutations
        record["complexity_stats"]["iteration$(iteration)"]["mutation_time"] = mutation_time
        record["complexity_stats"]["iteration$(iteration)"]["mutation_time_avg"] = mutation_time_avg
    end

    return (pop, best_examples_seen, num_evals)
end

function optimize_and_simplify_population(
    dataset::D, pop::P, options::AbstractOptions, curmaxsize::Int, record::RecordType, iteration=nothing
)::Tuple{P,Float64} where {T,L,D<:Dataset{T,L},P<:Population{T,L}}
    array_num_evals = zeros(Float64, pop.n)
    do_optimization = rand(pop.n) .< options.optimizer_probability
    # Note: we have to turn off this threading loop due to Enzyme, since we need
    # to manually allocate a new task with a larger stack for Enzyme.
    should_thread = !(options.deterministic) && !(isa(options.autodiff_backend, AutoEnzyme))
    should_thread = false

    batched_dataset = options.batching ? batch(dataset, options.batch_size) : dataset

    optimization_start_time = time()
    num_optimizations = 0

    @threads_if should_thread for j in 1:(pop.n)
        if options.should_simplify
            tree = pop.members[j].tree
            tree = simplify_tree!(tree, options.operators)
            tree = combine_operators(tree, options.operators)
            pop.members[j].tree = tree
        end
        if options.should_optimize_constants && do_optimization[j]
            # TODO: Might want to do full batch optimization here?
            pop.members[j], array_num_evals[j] = optimize_constants(
                batched_dataset, pop.members[j], options
            )
            num_optimizations += 1
        end
    end
    num_evals = sum(array_num_evals)
    pop, tmp_num_evals = finalize_costs(dataset, pop, options)
    num_evals += tmp_num_evals

    optimization_time = time() - optimization_start_time
    optimization_time_avg = optimization_time / num_optimizations

    # Now, we create new references for every member,
    # and optionally record which operations occurred.
    for j in 1:(pop.n)
        old_ref = pop.members[j].ref
        new_ref = generate_reference()
        pop.members[j].parent = old_ref
        pop.members[j].ref = new_ref
    end

    @recorder begin
        if iteration !== nothing
            record["complexity_stats"]["iteration$(iteration)"]["num_optimizations"] = num_optimizations
            record["complexity_stats"]["iteration$(iteration)"]["optimization_time"] = optimization_time
            record["complexity_stats"]["iteration$(iteration)"]["optimization_time_avg"] = optimization_time_avg
        end
    end

    return (pop, num_evals)
end

end
