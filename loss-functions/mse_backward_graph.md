# MSE Backward Pass Visualization

## Forward Pass Graph

```mermaid
graph LR
    subgraph Inputs
        yp0["y_pred[0]=1.0"]
        yp1["y_pred[1]=2.0"]
        yp2["y_pred[2]=3.0"]
        yt0["y_true[0]=1.5"]
        yt1["y_true[1]=2.0"]
        yt2["y_true[2]=2.5"]
    end

    subgraph "Diff (subtract)"
        d0["diff[0]=-0.5"]
        d1["diff[1]=0.0"]
        d2["diff[2]=0.5"]
    end

    subgraph "Square (^2)"
        sq0["sq[0]=0.25"]
        sq1["sq[1]=0.0"]
        sq2["sq[2]=0.25"]
    end

    subgraph "Sum"
        sum["sum=0.5"]
    end

    subgraph "Mean (/n)"
        loss["loss=0.1667"]
    end

    yp0 --> d0
    yt0 --> d0
    yp1 --> d1
    yt1 --> d1
    yp2 --> d2
    yt2 --> d2

    d0 --> sq0
    d1 --> sq1
    d2 --> sq2

    sq0 --> sum
    sq1 --> sum
    sq2 --> sum

    sum --> loss
```

## Backward Pass (gradients flow right-to-left)

```
loss.grad = 1.0
    ↓ (/n backward: multiply by 1/3)
sum.grad = 0.333
    ↓ (sum backward: distribute equally)
sq[0].grad = 0.333,  sq[1].grad = 0.333,  sq[2].grad = 0.333
    ↓ (square backward: multiply by 2*diff due to chain rule)
diff[0].grad = 0.333 * 2 * (-0.5) = -0.333
diff[1].grad = 0.333 * 2 * (0.0)  =  0.0
diff[2].grad = 0.333 * 2 * (0.5)  =  0.333
    ↓ (distribute backward alongwith sign multiplication: +1 to y_pred, -1 to y_true)
y_pred[0].grad = -0.333
y_pred[1].grad =  0.0
y_pred[2].grad =  0.333
```

## Summary

- **Forward**: `L = (1/n) * Σ(y_pred - y_true)²`
- **Backward**: `dL/dy_pred = (2/n) * (y_pred - y_true)`
