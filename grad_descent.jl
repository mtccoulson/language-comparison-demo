using CSV

#read in and prep the data
df = CSV.read("/Users/morleycoulson/Data Science/language-comparison-demo/fake_data.csv")
X = convert(Matrix, df[:,1:11])
y = convert(Vector, df[!, 12])

#create logistic regression functions
#note the periods are needed to specify that this is a vectorised operation
function compute_sigmoid(x)
    return 1 / (1 + exp(-x))
end

#note the ' indicates a transpose operation
function compute_log_loss(x, y)
    N = size(x)[1]
    return (1 / N) * sum(-y' * log.(x) - (1 .- y)' * log.(1 .- x))
end

function compute_gradient(X, P_x, y)
    N = size(X)[1]
    return (1/N) * X' * (P_x - y)
end

function fit_logistic_regression(X, y; eta=0.01, max_iter=500)
    params = zeros(size(X)[2])
    current_loss = 0
    N = size(X)[1]

    for i in range(1, stop = max_iter)
        P_x = compute_sigmoid.(X * params)
        current_loss = compute_log_loss(P_x, y)
        current_grad = compute_gradient(X, P_x, y)
        params = params - eta*current_grad
    end
    return params
end

theta = fit_logistic_regression(X, y)
preds = compute_sigmoid.(X * theta)

function compute_auc(;preds, truth)
    rank = sortperm(sortperm(preds))
    n_pos = sum(truth .== 1)
    n_neg = sum(truth .== 0)
    auc = (sum(rank[truth .== 1]) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    return auc
end

auc = compute_auc(preds=preds, truth=y)
println(auc)
