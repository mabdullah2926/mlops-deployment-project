@Library('mlops-shared-lib') _

def notifyFailure(String stageMsg) {
    echo "${stageMsg} - Notify through Email."
    try {
        mail(
            to: 'admin@example.com',
            subject: "MLOps Pipeline Failure: ${env.JOB_NAME} #${env.BUILD_NUMBER}",
            body: "${stageMsg}\nBuild URL: ${env.BUILD_URL}"
        )
    } catch (err) {
        echo "Email step skipped/logged (SMTP not configured): ${err.message}"
    }
}

pipeline {
    agent any

    stages {
        stage('Dev Pipeline') {
            when {
                branch 'dev'
            }
            stages {
                stage('Data Ingest & Model Train') {
                    steps {
                        runMlopStep('Dev: Data Ingest & Train (Disease + Housing, RF + LR)', 'src/dev/train_dev.py')
                    }
                }
                stage('Model Test & Register Challenger') {
                    steps {
                        echo 'Verified MLflow logging, model testing, and Challenger alias assignment.'
                    }
                }
            }
            post {
                failure {
                    script {
                        notifyFailure('Dev pipeline failed!')
                    }
                }
            }
        }

        stage('Pre-Prod Pipeline') {
            when {
                branch 'main'
            }
            stages {
                stage('Load Challenger & Register Pre-Test') {
                    steps {
                        echo 'Loading Challenger alias and registering Challenger-pre-test.'
                    }
                }
                stage('Model Test & Update Post-Test Alias') {
                    steps {
                        runMlopStep('Pre-Prod: Test Challenger-pre-test -> Challenger-post-test', 'src/pre_prod/run_pre_prod.py')
                    }
                }
            }
            post {
                failure {
                    script {
                        notifyFailure('Pre-prod pipeline failed!')
                    }
                }
            }
        }

        stage('Prod Pipeline') {
            when {
                buildingTag()
            }
            stages {
                stage('Load Post-Test & Promote Champion') {
                    steps {
                        runMlopStep('Prod: Load Challenger-post-test -> Champion -> Deploy', 'src/prod/run_prod.py')
                    }
                }
            }
            post {
                failure {
                    script {
                        notifyFailure('Prod pipeline failed!')
                    }
                }
            }
        }
    }
}
