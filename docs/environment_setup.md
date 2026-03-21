# Setup środowiska

Zakłada się, że użytkownik działa na systemie Ubuntu 24.04

## Setup dockera
W celu ułatwienia pracy nad projektem wykorzystana została konteneryzacja środowiska. 

1. Jeżeli użytkownik nie posiada zainstalowanego dockera należy tego dokonać z wykorzystaniem [instrukcji](https://docs.docker.com/engine/install/ubuntu/).
2. Po proprawnej instalacji zakończonej weryfikacją i przeładowaniem grup użytkowników należy zainstalować  NVIDIA  Container toolkit, który pozwoli na akcelerację GPU wewnątrz kontenera. Można skorzystać przy tym z [instrukcji](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
    1. Po instalacji należy dołączyć toolkit do dockera:
    ```bash
    sudo nvidia-ctk runtime configure --runtime=docker
    ```
    2. Zrestartować daemon Dockera:
    ```bash
    sudo systemctl restart docker
    ```
    3. Sprawdzić czy "nvidia" jest widoczna na liście Runtimes:
    ```
    docker info | grep -i -E 'Runtimes|Default Runtime'
    ```

## Setup przestrzeni roboczej
1. Utworzyć folder:
```bash
mkdir -p ~/SWIZ/src/
```
2. Pobrać repozytorium z kodami źródłowymi:
```bash
cd SWIZ/src/ && git clone git@github.com:VisteK528/SWIZ.git
```

3. Zbudować obraz:
```bash
docker compose build
```

4. Uruchomić kontener:
```bash
docker compose up
```
