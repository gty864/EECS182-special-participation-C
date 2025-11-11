import torch
from src.optim.sgd import sgd_step
from src.models.logistic import LogisticRegression

def test_momentum_accumulates():
    model = LogisticRegression()
    x = torch.randn(10, 2)
    y = torch.randint(0, 2, (10,)).float()
    opt = lambda: torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # manual
    velocity = {}
    for _ in range(2):
        model.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
        loss.backward()
        grads = [p.grad for p in model.parameters()]
        velocity = sgd_step(model.parameters(), grads, lr=0.1, momentum=0.9, velocity=velocity)

    # torch reference
    ref = LogisticRegression()
    ref_opt = opt()
    for _ in range(2):
        ref.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(ref(x), y)
        loss.backward()
        ref_opt.step()

    assert all(torch.allclose(p1, p2) for p1, p2 in zip(model.parameters(), ref.parameters()))