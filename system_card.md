# System Card for AutoDCA

| Version | Date       | Author                   | Changes            |
|:--------|:-----------|:-------------------------|:-------------------|
| 1.0     | 04.11.2025   | Oumou El Mouminine DHMINE             | Initial version    |
| 1.1     | 07.11.2025   | Oumou El Mouminine DHMINE             | Clarifying references to Equinor's internal processes and documentation.|

## About
It provides information about intended use and risk which may be useful to those using or assessing the system.
The structure of this card is loosely aligned with the requirements of the [EU AI Act Annex IV](https://artificialintelligenceact.eu/annex/4/) and [Article 11](https://artificialintelligenceact.eu/article/11/).

## Intended Use
**Purpose / Value Proposition:**  
AutoDCA (Automated Decline Curve Analysis) provides automation tools for efficient, high-quality decline curve analysis.

**Note:** DCA assumes wells exhibit steady production decline, typical of mature wells past peak production.
Deviations may impact prediction accuracy.
Use judgment and review debugging figures to assess performance.

**Primary Users:**  
- Reservoir Engineers
- Production Engineers
- Other professionals involved in well performance analysis and forecasting

**Limitations:**  
- Not suitable for wells in early production phase (unless it is declining)
- Requires sufficient historical data
  
## Equinor-Specific Supporting Documentation
- **CI Number:** Please use CI number 121455 to find associated documentation
## Risk
In the following tables, the *risk factor category* (*RC*) is taken from Equinor internal governing document RM100 R-105908 with the following abbreviations

| RC (Risk factor category)                       | abbreviation |
|:------------------------------------------------|:-------------|
| Human and Organisational                        | H            |
| Technical and Operational                       | T            |
| Communities and local environment               | C            |
| Politics, regulatory and security               | P            |
| Market, supply chain and business relationships | M            | 

### Opportunity (Upside Risk)
| RC | Description                                      | Consequence                                      | Stakeholders                  |
|:--:|:-------------------------------------------------|:-------------------------------------------------|:------------------------------|
| T  | More accurate, standardized DCA in forecasting for engineers | Improved decision-making, optimized production strategies | Reservoir Engineers, Production Engineers |
| H  | Increased job satisfaction for engineers due to reduced manual workload | Higher morale, better retention rates | Reservoir Engineers, Production Engineers |
| T  | Traceability of forecast inputs and logging of well forecast inputs | Enhanced auditability and traceability of forecasts  | Equinor-internal Product Governance |
| T  | Easier implementation and adoption due to integrating with existing commercial software (e.g., pForecast) | Reduced integration complexity | Reservoir Engineers, Production Engineers |
| T  | Scalable cloud architecture| Reduces computational bottlenecks |Equinor-internal Architecture Team|
| P  | Clear governance and architecture compliance (Equinor-Internal Processes: AC-1526, TRL4) | Smoother scaling and enterprise adoption | Equinor-internal Product Governance,Equinor-internal Architecture Team |
| T  | Generalization method could potentially reduce the sensitivity of the Decline Curve to outliers and noise in production data |  Improve decline curve fitting |  Reservoir Engineers, Production Engineers |

### Mitigated Risk
| RC | Description                                      | Consequence           | Mitigating Action                                      | Stakeholders |
|:--:|:-------------------------------------------------|:----------------------|:-------------------------------------------------------|:-------------|
| T  | Code maintainability       | Delays in bug fixes, difficulty adding features | Modular architecture; CI/CD with GHAS (CodeQL, Dependabot); governance via Equinor-internal Processes: AC-1526, TRL4|Equinor-internal IT Team, Equinor-internal Product Governance |
| P  | Non-compliance with subsurface data management standards | Regulatory violations, data governance issues, restricted data access | Adopt TR1119 Data Management of Subsurface Data standard; obtain Chief approval for data handling procedures; implement compliant data lineage tracking | Chief Geologist, Data Management, Legal |
| H  | Over-reliance on automated forecasts without validation | Poor business decisions, financial losses | Mandatory human review workflows; clear limitation documentation; SME validation requirements | Reservoir Engineers |
| T  | Data quality and availability issues in Production Data Mart | Inaccurate forecasts, system failures | Data validation pipelines; integration with PDM; fallback to manual workflows | Equinor-nternal PDM Team |
| T  | Dependency on 3rd Party PowerSim (Pforecast) for API Build  | Service interruptions, integration issues | Establish SLAs with 3rd party; implement fallback mechanisms; regular integration testing | Equinor-internal IT Team |
| T  | Inability to rollback to previous model and predictions | Prolonged outages, incorrect forecasts | Version control for models and configurations; automated rollback procedures | Equinor-internal IT Team |
| T  | Security: DSA App Registration & Subscription Keys | Unauthorized access, data breaches | Implement strict access controls; regular security audits; use of managed identities | Equinor-internal IT Security Team |
| T  | IT Operations Cost OverRun | Budget overruns, resource constraints | Implement cost monitoring and optimization strategies; regular budget reviews | Equinor-internal IT Team |


### Unmitigated Risk
| RC | Description                                      | Consequence           | Suggested Mitigation                                   | Stakeholders |
|:--:|:-------------------------------------------------|:----------------------|:-------------------------------------------------------|:-------------|
| T  | Model degradation over time due to changing reservoir conditions | Decreasing forecast accuracy | Users should re-run ADCA at least every six months |  Reservoir Engineering, Production Engineers |
| T  | Curve fitting may yield non-physical sense | Decreasing forecast accuracy | Validate model outputs against known reservoir behavior or reservoir simulators (Eclipse) results |Reservoir Engineers, Production Engineers|
## Human Oversight
- AutoDCA is **decision support** for production forecasting; engineers review automated curve fits and forecasts before making business decisions.
- **Required validation**: Subject Matter Expert (SME) review of decline curve parameters, forecast assumptions, and quality control plots.
- **Limitations**: Requires domain expertise to validate appropriateness of DCA for specific wells.
- **Anticipated issues**: Over-reliance on automated forecasts without proper quality control. 

## System Architecture
- **Type:** Python command-line tool (adca) and web application (AutoDCA App)
- **Backend Architecture:**
  - Kubernetes-hosted API services
  - Celery worker pods for job processing
  - Kubeflow pipelines for complex workflows
  - Azure Blob Storage for data persistence
- **Frontend**: React-based UI using Equinor Amplify Reach components
- **Integration Points**: 
  - Production Data Mart (PDM) for input data
  - pForecast for output consumption
- **Instructions**: 
    - For adca CLI: Configure wells via YAML files, execute analysis, review quality control plots, validate forecasts, export results
    - For AutoDCA App: Access PDM Data, configure analysis settings via UI, initiate analysis, review results

## Data Acquisition and Preparation
- **Primary Data Source**: Production Data Mart (PDM) containing well production time series data
- **Data Types**: Oil, gas, water and condensate production rates. On stream hours (time on)
- **Data Processing Pipeline**: Removal of missing values, consistency checks of data, etc 
- **Configuration Management**: YAML-based configuration files specifying well lists, parameters, and output requirements or config using the UI.
- **Data Security**: 
  - Internal Equinor use with appropriate access controls
  - Data encrypted in transit and at rest in Azure Blob Storage
  - Role-based access control aligned with PDM permissions
- **Backup Data Sources**: Local file uploads for offline analysis; manual data entry for validation scenarios

## Verification and Validation
- **Technical Verification**:
  - Unit testing for mathematical functions and data processing routines
  - Integration testing for end-to-end workflows
  - Security testing for data protection and access controls
- **Domain Expert Review**:
  - SME validation of curve fitting results across different reservoir types
  - Asset team validation

## Deployment
- **ADCA CLI Package**:
  - Distributed via internal GitHub repository
  - Installation via pip 
  - Compatible with Linux, Windows and Mac OS environments
  - Documentation and examples provided for self-service adoption
- **AutoDCA Web Application**:
  - Deployed on Equinor's AI Platform
  - Automated deployment pipeline using GitOps principles
  - Blue-green deployment strategy for zero-downtime updates
- **CI/CD Pipeline**:
  - GitHub Actions for automated testing and building
  - GitHub Advanced Security (GHAS) integration:
    - CodeQL for security vulnerability scanning
    - Dependabot for dependency management
    - Secret scanning for credential protection
  - Automated Docker image building and registry management
- **Infrastructure as Code**:
  - Helm charts for Kubernetes deployment configuration
  - Environment-specific configurations (dev, test, prod)
  - Monitoring and alerting setup via Prometheus and Grafana
- **Rollback Strategy**: Automated rollback capabilities; database migration rollback procedures; configuration version control
  
## Operations and Monitoring
- Monitor usage metrics.

## Post‑Market Phase Evaluation
- Review of adoption and feedback.

## Changes and Conformity
- Maintain change log in repos.
- Keep record of updates ready for potential yearly updates.
- Equinor-Internal Processes and Standards: AC‑1526, CSA, TRL guidance.
