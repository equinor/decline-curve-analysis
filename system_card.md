# System Card for AutoDCA

| Version | Date       | Author                   | Changes            |
|:--------|:-----------|:-------------------------|:-------------------|
| 1.0     | 10.10.2025   | Oumou El Mouminine DHMINE             | Initial version    |

<br>

## About
This document is a system card. It provides information about intended use and risk which may be useful to those using or assessing the system. The structure of this card is loosely aligned with the requirements of the [EU AI Act Annex IV](https://artificialintelligenceact.eu/annex/4/). and [Article 11](https://artificialintelligenceact.eu/article/11/).

## Intended Use
**Purpose / Value Proposition:**  
AutoDCA (Automated Decline Curve Analysis) provides automation tools for efficient, high-quality decline curve analysis. The system increases both efficiency and accuracy in forecasting where a decline curve approach is appropriate, offering significant improvements over traditional manual DCA methods.

**Note:** DCA assumes wells exhibit steady production decline, typical of mature wells past peak production. Deviations may impact prediction accuracy. Use engineering judgment and review ADCA debugging figures to assess performance.

**Primary Users:**  
- Reservoir Engineers
- Production Engineers
- Other professionals involved in well performance analysis and forecasting

**Operating Context:**  
- Internal-only Equinor use (IP and internal calibration data)
- Delivered as:
  1. **AutoDCA CLI**: Python Command-line tool
  2. **AutoDCA App**: Web-based application at scale for collaborative use with DSA Toolbox

**Limitations:**  
- Requires consistent, uninterrupted production decline
- Not suitable for wells in early production phase
- Requires sufficient historical data

<br>

## Supporting Documentation
- **Business inventory:** [Automated-Decline-Curve-Analysis in Service@Equinor](https://equinor.service-now.com/nav_to.do?uri=cmdb_ci_business_app.do?sys_id=1116e91f97a88a1c993f3077f053af78). 
- **Architecture contract:** [AC‑1526](https://org8b9f0d0d.crm19.dynamics.com/main.aspx?appid=9284c773-f710-ee11-9cbd-002248dcc582&pagetype=entityrecord&etn=ak_architecturecontract&id=4247220f-8849-4427-852e-fa7a9a4ba30c)
- **Colab:** [T23‑00305](https://colab.equinor.com/technologies/F891C23F-484A-4349-8111-3D8C464C8AA0/business-case)
- **Current TRL:** TRL 4

<br>

## Risk
In the following tables, the *risk factor category* (*RC*) is taken from [RM100 R-105908](https://aris.equinor.com/#default/item/c.requirement.Production.hholoN3kEeRmCQBQVrsUrw.-1) with the following abbreviations

| RC (Risk factor category)                       | abbreviation |
|:------------------------------------------------|:-------------|
| Human and Organisational                        | H            |
| Technical and Operational                       | T            |
| Communities and local environment               | C            |
| Politics, regulatory and security               | P            |
| Market, supply chain and business relationships | M            |

<br>

### Opportunity (Upside Risk)
| RC | Description                                      | Consequence                                      | Stakeholders                  |
|:--:|:-------------------------------------------------|:-------------------------------------------------|:------------------------------|
| T  | More accurate, standardized DCA in forecasting for engineers | Improved decision-making, optimized production strategies | Reservoir Engineers, Production Engineers |
| H  | Increased job satisfaction for engineers due to reduced manual workload | Higher morale, better retention rates | Reservoir Engineers, Production Engineers |
| T  | Traceability of forecast inputs and logging of well forecast inputs | Enhanced auditability and traceability of forecasts  | DSA, IT Ops, Product Governance |
| T  | Easier implementation and adoption due to integrating with existing commercial software (e.g., pForecast) | Reduced integration complexity | Reservoir Engineers, Production Engineers |
| T  | Scalable cloud architecture| Reduces computational bottlenecks | DSA, IT Ops, AI Platform teams |
| P  | Clear governance and architecture compliance (AC‑1526, TRL4) | Smoother scaling and enterprise adoption | Product governance, Architecture team, DSA |

<br>

### Mitigated Risk
| RC | Description                                      | Consequence           | Mitigating Action                                      | Stakeholders |
|:--:|:-------------------------------------------------|:----------------------|:-------------------------------------------------------|:-------------|
| T  | Code maintainability       | Delays in bug fixes, difficulty adding features | Modular architecture; CI/CD with GHAS (CodeQL, Dependabot); governance via AC‑1526 | DSA, IT Ops, Product Governance |
| P  | Non-compliance with subsurface data management standards | Regulatory violations, data governance issues, restricted data access | Adopt TR1119 Data Management of Subsurface Data standard; obtain Chief approval for data handling procedures; implement compliant data lineage tracking | Chief Geologist, Data Management, Legal |
| H  | Over-reliance on automated forecasts without validation | Poor business decisions, financial losses | Mandatory human review workflows; clear limitation documentation; SME validation requirements | Reservoir Engineers, Management |
| T  | Data quality and availability issues in Production Data Mart | Inaccurate forecasts, system failures | Data validation pipelines; integration with PDM; fallback to manual workflows | PDM Team, DSA |
| T  | Dependency on 3rd Party PowerSim (Pforecast) for API Build  | Service interruptions, integration issues | Establish SLAs with 3rd party; implement fallback mechanisms; regular integration testing | DSA, IT Ops |
| T  | Inability to rollback to previous model and predictions | Prolonged outages, incorrect forecasts | Version control for models and configurations; automated rollback procedures | DSA, IT Ops |
| T  | Security: DSA App Registration  & Subscription Keys | Unauthorized access, data breaches | Implement strict access controls; regular security audits; use of managed identities | IT Security, DSA |
| T  | IT Operations Cost OverRun | Budget overruns, resource constraints | Implement cost monitoring and optimization strategies; regular budget reviews | IT Ops, DSA |

<br>

### Unmitigated Risk
| RC | Description                                      | Consequence           | Suggested Mitigation                                   | Stakeholders |
|:--:|:-------------------------------------------------|:----------------------|:-------------------------------------------------------|:-------------|
| T  | Model degradation over time due to changing reservoir conditions | Decreasing forecast accuracy | Implement model monitoring; establish retraining protocols; version control for models | DSA, Reservoir Engineering |
<br>

## Human Oversight
- AutoDCA is **decision support** for production forecasting; engineers review automated curve fits and forecasts before making business decisions.
- **Required validation**: Subject Matter Expert (SME) review of decline curve parameters, forecast assumptions, and quality control plots.
- **Limitations**: False positives/negatives in curve fitting; requires domain expertise to validate appropriateness of decline curve methodology for specific wells.
- **Anticipated issues**: Over-reliance on automated forecasts without considering reservoir physics. 
- **Governance**: Clear escalation paths for disputed forecasts; mandatory peer review for high-impact predictions.
<br>

## System Architecture
- **Type:** Python command-line tool (AutoDCA CLI) and web application (AutoDCA App)
- **Delivery Modes:**
  - **AutoDCA CLI**: Python package for local execution on Mac/Linux/Windows systems
  - **AutoDCA App**: Web-based application deployed on Kubernetes AI Platform
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
    - For AutoDCA CLI: Configure wells via YAML files, execute analysis, review quality control plots, validate forecasts, export results
    - For AutoDCA App: Access PDM Data, configure analysis settings via UI, initiate analysis, review results
<br>

## Data Acquisition and Preparation
- **Primary Data Source**: Production Data Mart (PDM) containing well production time series data
- **Data Types**: Oil, gas, and water production rates; flowing and static pressures; well configuration data
- **Data Processing Pipeline**: Historical data aggregation and trend analysis
- **Configuration Management**: YAML-based configuration files specifying well lists, analysis parameters, and output requirements or config using the UI.
- **Data Security**: 
  - Internal Equinor use only with appropriate access controls
  - Data encrypted in transit and at rest in Azure Blob Storage
  - Role-based access control aligned with PDM permissions
- **Backup Data Sources**: Local file uploads for offline analysis; manual data entry for validation scenarios

<br>

## Verification and Validation
- **Technical Verification**:
  - Unit testing for mathematical functions and data processing routines
  - Integration testing for end-to-end workflows
  - Security testing for data protection and access controls
- **Domain Expert Review**:
  - SME validation of curve fitting results across different reservoir types
  - Asset team validation

<br>

## Deployment
- **ADCA CLI Package**:
  - Distributed via internal GitHub repository
  - Installation via pip 
  - Compatible with Linux and Windows environments
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

<br>

## Operations and Monitoring
- Monitor usage metrics.

<br>

## Post‑Market Phase Evaluation
- Review of adoption and feedback.

<br>

## Changes and Conformity
- Maintain change log in repos.
- Keep record of updates ready for potential yearly updates.
- Standards: AC‑1526, CSA, TRL guidance.

<br>

## Notes
- Contacts: Ashley Russell (Product Lead), Knut Utne Hollund, Tommy Odland, and Oumou El Mouminine Dhmine (Author).

