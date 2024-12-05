# 엔트로피 계산 함수
def calculate_entropy(probabilities):
    return -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)

# 훈련 함수 (엔트로피 기준 계산 포함)
def train_model_with_entropy(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # 훈련 단계
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 통계 계산
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = corrects.double() / total

        print(f"Train Loss: {epoch_loss:.4f} \n \t Acc: {epoch_acc:.4f}")

        # 검증 단계 (엔트로피 계산 포함)
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        all_entropies = []
        misclassified_entropies = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # 확률 계산 (Softmax 적용)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                entropies = calculate_entropy(probabilities)

                # 통계 계산
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                val_total += labels.size(0)

                # 엔트로피 저장
                all_entropies.extend(entropies)
                misclassified_entropies.extend(entropies[preds.cpu() != labels.cpu()])

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / val_total

        print(f"Validation Loss: {val_loss:.4f} \n \t Acc: {val_acc:.4f}\n")

        # 학습률 스케줄러 변경 (검증 성능 개선시만 적용)
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()
            scheduler.step()  # 성능 개선이 있을 때만 학습률을 변경

    # 엔트로피 기준 계산
    mean_entropy = np.mean(all_entropies)
    std_entropy = np.std(all_entropies)
    mean_misclassified_entropy = np.mean(misclassified_entropies)
    print(f"Mean Entropy (All): {mean_entropy:.4f}")
    print(f"Std Entropy (All): {std_entropy:.4f}")
    print(f"Mean Entropy (Misclassified): {mean_misclassified_entropy:.4f}")

    # 최종 엔트로피 기준 계산
    alpha = 0.5  # 하이퍼파라미터
    k = 1.0  # 표준편차 스케일
    final_threshold = alpha * mean_misclassified_entropy + (1 - alpha) * (mean_entropy + k * std_entropy)
    print(f"Final Threshold: {final_threshold:.4f}")

    # 최종적으로 학습한 가장 좋은 모델 가중치 로드
    model.load_state_dict(best_model_wts)
    return model, final_threshold

# 모델 학습
trained_model, entropy_threshold = train_model_with_entropy(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10)