pipeline {
    agent none
    options {
        buildDiscarder(logRotator(daysToKeepStr: '7', artifactDaysToKeepStr: '7'))
        timeout(time: 30, unit: 'MINUTES')
    }
    environment {
        SLACK_CHANNEL = '#jenkins-notifications'
    }
    parameters {
        booleanParam(defaultValue: true,
                     description: 'Send Slack notifications',
                     name: 'slackNotifications')
    }
    stages {
        stage('Build pyspatools') {
            agent {
                docker {
                    image "${env.NEXUS_DOCKER}/holoplot/infra/pypi-manager:0.0.95"
                    registryUrl env.NEXUS_DOCKER_URL
                    registryCredentialsId 'nexus'
                    label 'linux'
                }
            }
            stages {
                stage('Update version') {
                    steps {
                        script {
                            def version = readFile(file: 'VERSION')
                            if (env.BRANCH_NAME == 'main') {
                                version += ".${env.BUILDER_NUMBER}"

                            } else {
                                version += ".${env.BUILDER_NUMBER}+${BRANCH_NAME.replaceAll(/[^a-zA-Z\d]/, '.')}"
                            }
                            writeFile(file: 'VERSION', text: version)

                            def python_version = readFile(file: 'pyspatools/__version__.py')
                            python_version = python_version.replace('{{VERSION}}', version)
                            python_version = python_version.replace('{{REVISION}}', env.GIT_COMMIT)
                            writeFile(file: 'pyspatools/__version__.py', text: python_version)
                        }
                    }
                }

                stage('Build') {
                    steps {
                        script {
                            sh 'python3 -m build'
                        }
                    }
                }

                stage('Upload') {
                    steps {
                        script {
                            withCredentials([usernamePassword(
                                creadentialsId: 'nexus',
                                passwordVariable: 'NEXUS_PASSWORD',
                                usernameVariable: 'NEXUS_LOGIN'
                            )]) {
                                sh "twine upload --repository-url ${NEXUS_PYPI_UPLOAD_URL} -u ${NEXUS_LOGIN} -p ${NEXUS_PASSWORD} --verbose dist/*.whl"

                            }
                        }
                    }
                }
            }
            post {
                cleanup {
                        cleanWs()
                }
            }
        }
    }
    post {
        always {
            script {
                if (params.slackNotifications) {
                    if (currentBuild.result == 'SUCCESS') {
                        slackSend (color: '#00FF00',
                                message: "SUCCESS: <${env.BUILD_URL} | ${currentBuild.fullDisplayName}>",
                                channel: env.SLACK_CHANNEL)
                    } else {
                        slackSend (color: '#FF0000',
                                message: "FAIL: <${env.BUILD_URL} | ${currentBuild.fullDisplayName}>",
                                channel: env.SLACK_CHANNEL)
                    }
                }
            }
        }

}
