module CheckConstraintsModule

using DynamicExpressions:
    AbstractExpressionNode,
    AbstractExpression,
    get_tree,
    count_depth,
    tree_mapreduce,
    get_child,
    get_variable_names
using ..CoreModule: AbstractOptions
using ..ComplexityModule: compute_complexity, past_complexity_limit

using ..MultiFeatureNodeModule

# Generic operator complexity checking for any degree
function flag_operator_complexity(
    tree::AbstractExpressionNode, degree::Int, op::Int, cons, options::AbstractOptions
)::Bool
    return any(tree) do subtree
        subtree.degree == degree &&
            subtree.op == op &&
            _check_operator_constraints(subtree, degree, cons, options)
    end
end

function _check_operator_constraints(
    node::AbstractExpressionNode, degree::Int, cons, options::AbstractOptions
)::Bool
    @assert degree != 0

    return any(1:degree) do i
        cons[i] != -1 && past_complexity_limit(get_child(node, i), options, cons[i])
    end
end

# function count_max_nestedness(tree, degree, op)
#     # TODO: Update this to correctly share nodes
#     nestedness = tree_mapreduce(
#         t -> 0,  # Leafs
#         t -> (t.degree == degree && t.op == op) ? 1 : 0,  # Branches
#         (p, c...) -> p + max(c...),  # Reduce
#         tree;
#         break_sharing=Val(true),
#     )
#     # Remove count of self:
#     is_self = tree.degree == degree && tree.op == op
#     return nestedness - (is_self ? 1 : 0)
# end
function count_max_nestedness(tree, degree, op)
    tree.degree == 0 && return 0
    tree.degree == 1 && return tree.l.degree == degree && tree.l.op == op
    return Int(tree.l.degree == degree && tree.l.op == op) + Int(tree.r.degree == degree && tree.r.op == op)
end

"""Check if there are any illegal combinations of operators"""
function flag_illegal_nests(tree::AbstractExpressionNode, options::AbstractOptions)::Bool
    # We search from the top first, then from child nodes at end.
    nested_constraints = options.nested_constraints
    isnothing(nested_constraints) && return false
    any(tree) do subtree
        any(nested_constraints) do (degree, op_idx, op_constraints)
            subtree.degree == degree &&
                subtree.op == op_idx &&
                any(op_constraints) do (nested_degree, nested_op_idx, max_nestedness)
                    count_max_nestedness(subtree, nested_degree, nested_op_idx) >
                    max_nestedness
                end
        end
    end
end

function count_features(
    tree::AbstractExpressionNode,
    feature::String,
    variable_names::AbstractVector{<:AbstractString}
)::Int
    return tree_mapreduce(
        let vn = variable_names
            t -> (!t.constant && occursin(feature, vn[t.feature])) ? 1 : 0
        end,
        t -> 0,
        +,
        tree,
        break_sharing=Val(true),
    )
end
function count_features(
    tree::MultiFeatureNode,
    feature::String,
    variable_names::AbstractVector{<:AbstractString}
)::Int
    return tree_mapreduce(
        let vn = variable_names
            # t -> (!t.constant && t.is_single_feature && vn[t.feature] == feature) ? 1 : 0
            t -> (t.constant ? 0 : (
                    t.is_single_feature ? vn[t.feature] == feature : (
                        any(t.features) do f
                            startswith(vn[f], feature)
                        end
                    )
                )
            )
        end,
        t -> 0,
        +,
        tree,
        break_sharing=Val(true),
    )
end

function count_constants(
    tree::AbstractExpressionNode
)::Int
    # return tree_mapreduce(
    #     t -> t.constant ? 1 : 0,
    #     t -> 0,
    #     +,
    #     tree,
    #     break_sharing=Val(true),
    # )
    return tree.constant ? 1 : 0
end

function flag_illegal_feature_constraints(
    tree::AbstractExpressionNode,
    options::AbstractOptions,
    variable_names::AbstractVector{<:AbstractString}
)::Bool
    unary_feature_constraints = hasproperty(options, :unary_feature_constraints) ? options.unary_feature_constraints : nothing
    binary_feature_constraints = hasproperty(options, :binary_feature_constraints) ? options.binary_feature_constraints : nothing
    # isnothing(unary_feature_constraints) && return false
    # isnothing(binary_feature_constraints) && return false
    any(tree) do subtree
        # Check unary constraints
        unary_violated = isnothing(unary_feature_constraints) ? false : any(unary_feature_constraints) do (op_idx, op_constraints)
            subtree.degree == 1 &&
            subtree.op == op_idx &&
            any(op_constraints) do (feature, max_count)
                count_features(subtree, feature, variable_names) > max_count
            end
        end
        
        # Check binary constraints  
        binary_violated = isnothing(binary_feature_constraints) ? false : any(binary_feature_constraints) do (op_idx, op_constraints)
            subtree.degree == 2 &&
            subtree.op == op_idx &&
            any(op_constraints) do (feature, (max_count_left, max_count_right))
                count_features(subtree.l, feature, variable_names) > max_count_left ||
                count_features(subtree.r, feature, variable_names) > max_count_right
            end
        end
        
        # Return true if ANY constraints are violated
        return unary_violated || binary_violated
    end
end

function flag_illegal_constant_constraints(
    tree::AbstractExpressionNode,
    options::AbstractOptions
)::Bool
    unary_constant_constraints = hasproperty(options, :unary_constant_constraints) ? options.unary_constant_constraints : nothing
    binary_constant_constraints = hasproperty(options, :binary_constant_constraints) ? options.binary_constant_constraints : nothing
    # isnothing(unary_feature_constraints) && return false
    # isnothing(binary_feature_constraints) && return false
    any(tree) do subtree
        # Check unary constraints
        unary_violated = isnothing(unary_constant_constraints) ? false : any(unary_constant_constraints) do (op_idx, max_count)
            subtree.degree == 1 &&
            subtree.op == op_idx &&
            count_constants(subtree.l) > max_count
        end
        
        # Check binary constraints  
        binary_violated = isnothing(binary_constant_constraints) ? false : any(binary_constant_constraints) do (op_idx, (max_count_left, max_count_right))
            subtree.degree == 2 &&
            subtree.op == op_idx &&
            (count_constants(subtree.l) > max_count_left ||
             count_constants(subtree.r) > max_count_right)
        end
        
        # Return true if ANY constraints are violated
        return unary_violated || binary_violated
    end
end

"""Check if user-passed constraints are satisfied. Returns false otherwise."""
function check_constraints(
    ex::AbstractExpression,
    options::AbstractOptions,
    maxsize::Int,
    cached_size::Union{Int,Nothing}=nothing,
)::Bool
    tree = get_tree(ex)
    if get_variable_names(ex) !== nothing
        flag_illegal_feature_constraints(tree, options, get_variable_names(ex)) && return false
    end
    return check_constraints(tree, options, maxsize, cached_size)
end
function check_constraints(
    tree::AbstractExpressionNode,
    options::AbstractOptions,
    maxsize::Int,
    cached_size::Union{Int,Nothing}=nothing,
)::Bool
    @something(cached_size, compute_complexity(tree, options)) > maxsize && return false
    count_depth(tree) > options.maxdepth && return false
    any_invalid = any(enumerate(options.op_constraints)) do (degree, degree_constraints)
        any(enumerate(degree_constraints)) do (op_idx, cons)
            any(!=(-1), cons) &&
                flag_operator_complexity(tree, degree, op_idx, cons, options)
        end
    end
    any_invalid && return false
    flag_illegal_nests(tree, options) && return false
    flag_illegal_constant_constraints(tree, options) && return false
    return true
end

check_constraints(ex::Union{AbstractExpression,AbstractExpressionNode}, options::AbstractOptions)::Bool = check_constraints(
    ex, options, options.maxsize
)

end
