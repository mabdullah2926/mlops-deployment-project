@Library('mlops-shared-lib') _

pipeline {
    agent any

    stages {
        stage('Dev Pipeline') {
            when {
                branch 'dev'
            }
            steps {
                runMlopStep('Dev Pipeline (Ingest -> Train -> Test -> Challenger)', 'src/dev/train_dev.py')
            }
            post {
                failure {
                    echo "Dev pipeline failed! Notify through Email."
                }
            }
        }

        stage('Pre-Prod Pipeline') {
            when {
                branch 'main'
            }
            steps {
                runMlopStep('Pre-Prod Pipeline (Challenger -> Challenger-pre-test -> Test -> Challenger-post-test)', 'src/pre_prod/run_pre_prod.py')
            }
            post {
                failure {
                    echo "Pre-prod pipeline failed! Notify through Email."
                }
            }
        }

        stage('Prod Pipeline') {
            when {
                buildingTag()
            }
            steps {
                runMlopStep('Prod Pipeline (Challenger-post-test -> Champion -> Deploy)', 'src/prod/run_prod.py')
            }
            post {
                failure {
                    echo "Prod pipeline failed! Notify through Email."
                }
            }
        }
    }
}
