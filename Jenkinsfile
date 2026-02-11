pipeline {
    agent any

    triggers {
        // Run daily at 6 AM
        cron('0 6 * * *')
    }

    environment {
        PROJECT_DIR = '/Users/danielmenesesleon/PycharmProjects/LSTM_Forecast'
        VENV_PATH = "${PROJECT_DIR}/venv/bin/activate"
    }

    stages {
        stage('Notify Start') {
            steps {
                dir("${PROJECT_DIR}") {
                    sh """
                        source ${VENV_PATH}
                        python notify_pipeline.py \
                            --status started \
                            --job "${env.JOB_NAME}" \
                            --build "${env.BUILD_NUMBER}" \
                            --branch "${env.GIT_BRANCH ?: 'main'}" \
                            --url "${env.BUILD_URL}"
                    """
                }
            }
        }

        stage('Setup') {
            steps {
                dir("${PROJECT_DIR}") {
                    sh """
                        source ${VENV_PATH}
                        pip install -r requirements.txt --quiet
                    """
                }
            }
        }

        stage('Run Forecast') {
            steps {
                dir("${PROJECT_DIR}") {
                    sh """
                        source ${VENV_PATH}
                        python main.py
                    """
                }
            }
        }

        stage('DS Audit') {
            when {
                // Run audit only on Mondays
                expression { 
                    return new Date().format('u') == '1' || env.RUN_AUDIT == 'true'
                }
            }
            steps {
                dir("${PROJECT_DIR}") {
                    sh './run_ds_audit.sh'
                }
            }
        }

        stage('Archive Results') {
            steps {
                dir("${PROJECT_DIR}") {
                    archiveArtifacts artifacts: 'images/*.png', allowEmptyArchive: true
                    archiveArtifacts artifacts: 'audit_reports/*.md', allowEmptyArchive: true
                }
            }
        }
    }

    post {
        success {
            dir("${PROJECT_DIR}") {
                sh """
                    source ${VENV_PATH}
                    python notify_pipeline.py \
                        --status success \
                        --job "${env.JOB_NAME}" \
                        --build "${env.BUILD_NUMBER}" \
                        --branch "${env.GIT_BRANCH ?: 'main'}" \
                        --duration "${currentBuild.durationString}"
                """
            }
        }

        failure {
            dir("${PROJECT_DIR}") {
                script {
                    // Capture the failed stage and error
                    def failedStage = env.STAGE_NAME ?: 'Unknown'
                    def errorMsg = currentBuild.description ?: 'Check console output for details'
                    
                    sh """
                        source ${VENV_PATH}
                        python notify_pipeline.py \
                            --status failure \
                            --job "${env.JOB_NAME}" \
                            --build "${env.BUILD_NUMBER}" \
                            --branch "${env.GIT_BRANCH ?: 'main'}" \
                            --stage "${failedStage}" \
                            --error "${errorMsg}" \
                            --url "${env.BUILD_URL}"
                    """
                }
            }
        }

        aborted {
            dir("${PROJECT_DIR}") {
                sh """
                    source ${VENV_PATH}
                    python notify_pipeline.py \
                        --status failure \
                        --job "${env.JOB_NAME}" \
                        --build "${env.BUILD_NUMBER}" \
                        --error "Pipeline was aborted by user" \
                        --url "${env.BUILD_URL}"
                """
            }
        }
    }
}
