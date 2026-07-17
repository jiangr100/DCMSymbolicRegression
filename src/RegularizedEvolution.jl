module RegularizedEvolutionModule

using DynamicExpressions: string_tree
using ..CoreModule: AbstractOptions, Dataset, RecordType, DATA_TYPE, LOSS_TYPE
using ..PopulationModule: Population, best_of_sample
using ..PopMemberModule: PopMember
using ..AdaptiveParsimonyModule: RunningSearchStatistics
using ..MutateModule: next_generation, crossover_generation
using ..RecorderModule: @recorder
using ..UtilsModule: argmin_fast, argmax_fast

# Pass through the population several times, replacing the oldest
# with the fittest of a small subsample
function reg_evol_cycle(
    dataset::Dataset{T,L},
    pop::P,
    temperature,
    curmaxsize::Int,
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
    record::RecordType,
)::Tuple{Vector{PopMember},Float64} where {T<:DATA_TYPE,L<:LOSS_TYPE,P<:Population{T,L}}    # Tuple{P, Float64}
    num_evals = 0.0
    n_evol_cycles = ceil(Int, pop.n / options.tournament_selection_n)
    new_pop = PopMember[]

    for i in 1:n_evol_cycles
        if rand() > options.crossover_probability
            allstar = best_of_sample(pop, running_search_statistics, options)
            mutation_recorder = RecordType()
            baby, mutation_accepted, tmp_num_evals = next_generation(
                dataset,
                allstar,
                temperature,
                curmaxsize,
                running_search_statistics,
                options;
                tmp_recorder=mutation_recorder,
                population_for_backsolve=pop,
            )
            num_evals += tmp_num_evals

            if !mutation_accepted && options.skip_mutation_failures
                # Skip this mutation rather than replacing oldest member with unchanged member
                continue
            end

            push!(new_pop, baby)
        else # Crossover
            allstar1 = best_of_sample(pop, running_search_statistics, options)
            allstar2 = best_of_sample(pop, running_search_statistics, options)

            crossover_recorder = RecordType()
            baby1, baby2, crossover_accepted, tmp_num_evals = crossover_generation(
                allstar1,
                allstar2,
                dataset,
                curmaxsize,
                options;
                recorder=crossover_recorder,
            )
            num_evals += tmp_num_evals

            if !crossover_accepted && options.skip_mutation_failures
                continue
            end

            push!(new_pop, baby1)
            # A crossover operator may yield a single offspring (e.g. LLM semantic
            # crossover); it returns `nothing` in the second slot to signal that.
            isnothing(baby2) || push!(new_pop, baby2)
        end
    end

    return (new_pop, num_evals)
end

end
