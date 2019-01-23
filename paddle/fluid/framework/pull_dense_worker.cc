
int PullDenseWorker::Start() {
    _running = true;
    _t = std::thread(&DensePullThread::Run, this);
    return 0;
}

void PullDenseWorker::Run() {
    while (_running) {
        _pull_dense_status.resize(0);
        for (auto& t : ) {
            
        }
        if (_pull_dense_status.size() != 0) {
            WaitAll();
        }
        
        usleep(_sleep_time_ms);
    }
}

bool PullDenseWorker::CheckUpdateParam(uint64_t table_id) {
    std::lock_guard<std::mutex> lock(_mutex_for_version);
    auto& version = _training_versions[table_id];
    _current_version[table_id] = *(std::min_element(version.begin(),
                                                    version.end()));
    if (_current_version[table_id] - _last_versions[table_id] < _threshold) {
        return false;
    }
    return true;
}

void PullDenseWorker::ResetThreadVersion(uint64_t table_id) {
    std::lock_guard<std::mutex> lock(_mutex_for_version);
    _last_versions[table_id] = _current_version[table_id];
}

